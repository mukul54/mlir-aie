#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

"""
Mamba Selective Scan on AIE

This example implements the Mamba selective scan operation on a single AIE core.
The selective scan is the core of the Mamba architecture for sequence modeling.

State update formula:
    state[n] = state[n] * exp(dt[t] * A[n]) + B[t,n] * x[t] * dt[t]
    y[t] = sum_n(state[n] * C[t,n]) + D * x[t]

Where:
    x: [seq_len] - Hidden states (input)
    dt: [seq_len] - Timesteps (controls state decay)
    A: [d_state] - State transition matrix
    B: [seq_len, d_state] - Input projection
    C: [seq_len, d_state] - Output projection
    D: scalar - Skip connection
    state: [d_state] - SSM state (updated in-place)
    out: [seq_len] - Output
"""

import argparse
import numpy as np
import sys

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.iron.dtype import str_to_dtype


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Mamba Selective Scan Design",
        description="Emits MLIR code for Mamba selective scan on AIE",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu2")
    argparser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    argparser.add_argument("--d_state", type=int, default=16, help="State dimension")
    argparser.add_argument(
        "--dtype", type=str, choices=["f32", "bf16"], default="bf16"
    )
    args = argparser.parse_args()

    mamba_scan_design(
        args.dev,
        args.seq_len,
        args.d_state,
        args.dtype,
    )


def mamba_scan_design(dev, seq_len, d_state, dtype_str):
    """Generate MLIR for Mamba selective scan"""

    # Validate parameters
    assert seq_len > 0, "seq_len must be positive"
    assert d_state in [16], "Only d_state=16 currently supported"
    assert seq_len <= 512, "seq_len must be <= 512 for buffer sizes"

    dtype = str_to_dtype(dtype_str)

    # Buffer sizes
    # NPU1 shim tile has only 2 S2MM (input to array) DMA channels.
    # We must combine all inputs into at most 2 ObjectFIFOs from shim.
    #
    # Combined buffer 1: x + dt + A + D (small vectors)
    # Combined buffer 2: B + C (large projection matrices)
    x_sz = seq_len
    dt_sz = seq_len
    A_sz = d_state
    D_sz = 2  # Minimum 32-bit (2x bf16) transfer required for DMA

    # Combined buffer 1: [x..., dt..., A..., D, padding]
    x_dt_A_D_sz = x_sz + dt_sz + A_sz + D_sz

    B_sz = seq_len * d_state
    C_sz = seq_len * d_state
    B_C_sz = B_sz + C_sz  # Combined: [B flatten..., C flatten...]

    state_sz = d_state
    out_sz = seq_len

    with mlir_mod_ctx() as ctx:

        if dev == "npu":
            dev_ty = AIEDevice.npu1_1col
        else:
            dev_ty = AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            # Define buffer types - using combined buffers to reduce DMA channels
            # NPU1 shim tile has only 2 S2MM (input) DMA channels, so we need
            # to pack all inputs into 2 buffers maximum.
            #
            # Buffer 1: x + dt + A + D  (small vectors, combined)
            # Buffer 2: B + C           (large projection matrices)

            # Combined buffer type: [x..., dt..., A..., D, padding]
            x_dt_A_D_ty = np.ndarray[(x_dt_A_D_sz,), np.dtype[dtype]]

            # Combined B+C projection matrices
            B_C_ty = np.ndarray[(B_C_sz,), np.dtype[dtype]]

            state_ty = np.ndarray[(d_state,), np.dtype[dtype]]
            out_ty = np.ndarray[(seq_len,), np.dtype[dtype]]

            # Declare external kernel function
            # The kernel receives the combined buffer and extracts components internally
            zero_func = external_func(f"zero_{dtype_str}", inputs=[state_ty, T.i32()])
            mamba_scan_func = external_func(
                f"mamba_scan_{dtype_str}",
                inputs=[
                    x_dt_A_D_ty,  # Combined buffer: [x..., dt..., A..., D, pad]
                    B_C_ty,       # B_C (combined buffer: [B..., C...])
                    state_ty,     # state (in/out)
                    out_ty,       # out
                    T.i32(),      # seq_len
                    T.i32(),      # d_state
                ],
            )

            # Tile declarations
            shim_tile = tile(0, 0)
            mem_tile = tile(0, 1)
            compute_tile = tile(0, 2)

            # ================================================================
            # ObjectFIFOs for data movement
            # Using combined buffers to stay within 2 input DMA channels:
            # - 1 input ObjectFIFO for x+dt+A+D (small data, through mem tile)
            # - 1 input ObjectFIFO for B+C (large matrices, through mem tile)
            # - 1 output ObjectFIFO for result
            # ================================================================

            # Input 1: x_dt_A_D (combined x, dt, A, D) through mem tile
            inParams = object_fifo("inParams", shim_tile, mem_tile, 2, x_dt_A_D_ty)
            memParams = object_fifo("memParams", mem_tile, compute_tile, 2, x_dt_A_D_ty)
            object_fifo_link(inParams, memParams)

            # Input 2: B_C (combined B and C projection matrices) through mem tile
            inB_C = object_fifo("inB_C", shim_tile, mem_tile, 2, B_C_ty)
            memB_C = object_fifo("memB_C", mem_tile, compute_tile, 2, B_C_ty)
            object_fifo_link(inB_C, memB_C)

            # State buffer (local buffer on compute tile, not an ObjectFIFO)
            # Using buffer() creates a local memory allocation that doesn't need acquire/release
            state_buffer = buffer(
                tile=compute_tile,
                datatype=state_ty,
                name="state_buffer",
                initial_value=np.zeros(d_state, dtype=dtype),
            )

            # Output
            memOut = object_fifo("memOut", compute_tile, mem_tile, 2, out_ty)
            outY = object_fifo("outY", mem_tile, shim_tile, 2, out_ty)
            object_fifo_link(memOut, outY)

            # ================================================================
            # Core program
            # ================================================================

            @core(compute_tile, "kernels.a", stack_size=0x1000)
            def core_body():
                # Infinite loop pattern (like vector_scalar_add_placed.py)
                for _ in range_(sys.maxsize):
                    # Acquire inputs from ObjectFIFOs
                    elem_params = memParams.acquire(ObjectFifoPort.Consume, 1)
                    elem_B_C = memB_C.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = memOut.acquire(ObjectFifoPort.Produce, 1)

                    # Initialize state to zero on first iteration
                    # State buffer is a local buffer, accessed directly
                    zero_func(state_buffer, d_state)

                    # Run Mamba selective scan
                    # Kernel will extract x, dt, A, D from elem_params
                    # and B, C from elem_B_C
                    mamba_scan_func(
                        elem_params,
                        elem_B_C,
                        state_buffer,  # Local buffer, no acquire needed
                        elem_out,
                        seq_len,
                        d_state,
                    )

                    # Release ObjectFIFO buffers only
                    memParams.release(ObjectFifoPort.Consume, 1)
                    memB_C.release(ObjectFifoPort.Consume, 1)
                    memOut.release(ObjectFifoPort.Produce, 1)

            # ================================================================
            # Runtime sequence (host control)
            # ================================================================

            @runtime_sequence(
                np.ndarray[(x_dt_A_D_sz,), np.dtype[dtype]],  # Params (x+dt+A+D combined)
                np.ndarray[(B_C_sz,), np.dtype[dtype]],       # B_C (combined)
                np.ndarray[(out_sz,), np.dtype[dtype]],       # Out
            )
            def sequence(Params, B_C, Out):
                # Use shim_dma_single_bd_task pattern (like vector_scalar_add_placed.py)
                # This properly handles ObjectFIFO links through mem_tile
                in_params_task = shim_dma_single_bd_task(
                    inParams, Params, sizes=[1, 1, 1, x_dt_A_D_sz], issue_token=True
                )
                in_B_C_task = shim_dma_single_bd_task(
                    inB_C, B_C, sizes=[1, 1, 1, B_C_sz], issue_token=True
                )
                out_task = shim_dma_single_bd_task(
                    outY, Out, sizes=[1, 1, 1, seq_len], issue_token=True
                )

                # Start all DMAs
                dma_start_task(in_params_task, in_B_C_task, out_task)
                # Wait for all to complete
                dma_await_task(in_params_task, in_B_C_task, out_task)

    print(ctx.module)


if __name__ == "__main__":
    main()
