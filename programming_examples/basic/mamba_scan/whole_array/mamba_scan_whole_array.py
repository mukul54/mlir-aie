#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

"""
Mamba Selective Scan on AIE - Whole Array Design

Processes multiple independent channels in parallel across the full AIE array.
Each compute tile runs one channel of the selective scan independently:

    state[n] = state[n] * exp(dt[t] * A[n]) + B[t,n] * x[t] * dt[t]
    y[t] = sum_n(state[n] * C[t,n]) + D * x[t]

Data distribution strategy:
    - Per-channel inputs (x, dt, A, D) are DISTRIBUTED across tiles
    - Shared inputs (B, C) are BROADCAST to all tiles
    - Per-channel outputs are JOINED back to host

Architecture:
    NPU1: up to 4 columns x 4 rows = 16 channels
    NPU2: up to 8 columns x 4 rows = 32 channels

Host buffer layout:
    all_params: [col0_row0_params, col0_row1_params, ..., colN_row3_params]
    B_C:        [B_flat, C_flat]  (shared across all channels)
    all_out:    [col0_row0_out, col0_row1_out, ..., colN_row3_out]
"""

import argparse
import numpy as np
import sys

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_
from aie.iron.dtype import str_to_dtype


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Mamba Selective Scan - Whole Array",
        description="Emits MLIR code for Mamba selective scan on full AIE array",
    )
    argparser.add_argument(
        "--dev", type=str, choices=["npu", "npu2"], default="npu2"
    )
    argparser.add_argument(
        "--seq_len", type=int, default=128, help="Sequence length"
    )
    argparser.add_argument(
        "--d_state", type=int, default=16, help="State dimension"
    )
    argparser.add_argument(
        "--dtype", type=str, choices=["f32", "bf16"], default="bf16"
    )
    argparser.add_argument(
        "--n-aie-cols",
        type=int,
        choices=[1, 2, 4, 8],
        default=8,
        help="Number of AIE columns to use",
    )
    argparser.add_argument(
        "--channels-per-tile",
        type=int,
        default=1,
        help="Channels processed per compute tile (interleaved to hide scan latency)",
    )
    args = argparser.parse_args()

    mamba_scan_whole_array(
        args.dev,
        args.seq_len,
        args.d_state,
        args.dtype,
        args.n_aie_cols,
        args.channels_per_tile,
    )


def mamba_scan_whole_array(dev, seq_len, d_state, dtype_str, n_aie_cols, cpt=1):
    """Generate MLIR for whole-array Mamba selective scan.
    cpt = channels per compute tile; the kernel interleaves them so independent
    channels pipeline and hide the per-channel recurrence latency."""

    n_aie_rows = 4
    n_cores = n_aie_cols * n_aie_rows
    n_channels = n_cores * cpt          # cpt channels interleaved per compute tile

    # Validate parameters
    assert seq_len > 0, "seq_len must be positive"
    assert d_state in [16], "Only d_state=16 currently supported"
    assert seq_len <= 512, "seq_len must be <= 512 for buffer sizes"

    if dev == "npu" and n_aie_cols > 4:
        raise ValueError("NPU1 (Phoenix/Hawk) has at most 4 columns")
    if dev == "npu2" and n_aie_cols > 8:
        raise ValueError("NPU2 (Strix/Krackan) has at most 8 columns")

    dtype = str_to_dtype(dtype_str)

    # Per-CORE buffer sizes (cpt channels, channel-major to match mamba_scan_multi_bf16:
    #   [x: cpt*seq, dt: cpt*seq, A: cpt*d_state, D: cpt, pad])
    _base = 2 * cpt * seq_len + cpt * d_state + cpt
    params_per_channel = _base + ((4 - _base % 4) % 4)   # per-core block, pad to mult of 4

    B_sz = seq_len * d_state
    C_sz = seq_len * d_state
    B_C_sz = B_sz + C_sz

    out_per_channel = cpt * seq_len     # cpt channels' outputs per core

    # Packed sizes for L2 (one column = n_aie_rows channels)
    params_per_col = n_aie_rows * params_per_channel
    out_per_col = n_aie_rows * out_per_channel

    with mlir_mod_ctx() as ctx:

        if dev == "npu":
            if n_aie_cols == 1:
                dev_ty = AIEDevice.npu1_1col
            elif n_aie_cols == 2:
                dev_ty = AIEDevice.npu1_2col
            elif n_aie_cols == 4:
                dev_ty = AIEDevice.npu1
            else:
                raise ValueError(f"Unsupported n_aie_cols={n_aie_cols} for NPU1")
        else:
            dev_ty = AIEDevice.npu2

        @device(dev_ty)
        def device_body():

            # ============================================================
            # Buffer types
            # ============================================================

            # L1 types (per-core, cpt channels interleaved)
            params_l1_ty = np.ndarray[(params_per_channel,), np.dtype[dtype]]
            BC_ty = np.ndarray[(B_C_sz,), np.dtype[dtype]]
            state_ty = np.ndarray[(cpt * d_state,), np.dtype[dtype]]
            out_l1_ty = np.ndarray[(out_per_channel,), np.dtype[dtype]]

            # L2 types (per-column, packed for all rows)
            params_l2_ty = np.ndarray[(params_per_col,), np.dtype[dtype]]
            out_l2_ty = np.ndarray[(out_per_col,), np.dtype[dtype]]

            # ============================================================
            # Kernel function declarations (same kernel as single-core)
            # ============================================================

            zero_func = external_func(
                f"zero_{dtype_str}", inputs=[state_ty, T.i32()]
            )
            mamba_scan_func = external_func(
                f"mamba_scan_multi_{dtype_str}",
                inputs=[
                    params_l1_ty,  # [x,dt,A,D] for cpt channels (channel-major)
                    BC_ty,         # B+C combined (shared across channels)
                    state_ty,      # state (in/out) for cpt channels
                    out_l1_ty,     # output for cpt channels
                    T.i32(),       # seq_len
                    T.i32(),       # d_state
                    T.i32(),       # n_ch (channels per tile)
                ],
            )

            # ============================================================
            # Tile declarations
            # ============================================================

            tiles = [
                [tile(col, row) for col in range(n_aie_cols)]
                for row in range(6)
            ]
            shim_tiles = tiles[0]   # Row 0: host interface
            mem_tiles = tiles[1]    # Row 1: L2 memory
            core_tiles = tiles[2:]  # Rows 2-5: compute (4 rows)

            fifo_depth = 2

            # ============================================================
            # ObjectFIFO declarations
            # ============================================================

            params_l3l2 = [None] * n_aie_cols
            params_l2l1 = [[None] * n_aie_cols for _ in range(n_aie_rows)]

            BC_l3l2 = [None] * n_aie_cols
            BC_l2l1 = [None] * n_aie_cols

            out_l1l2 = [[None] * n_aie_cols for _ in range(n_aie_rows)]
            out_l2l3 = [None] * n_aie_cols

            for col in range(n_aie_cols):

                # --- Params: distribute from shim -> mem -> individual cores ---

                # L3 -> L2: packed params for all rows in this column
                params_l3l2[col] = object_fifo(
                    f"params_L3L2_{col}",
                    shim_tiles[col],
                    mem_tiles[col],
                    fifo_depth,
                    params_l2_ty,
                )

                # L2 -> L1: one FIFO per core (each gets its own channel)
                for row in range(n_aie_rows):
                    params_l2l1[row][col] = object_fifo(
                        f"params_L2L1_{col}_{row}",
                        mem_tiles[col],
                        core_tiles[row][col],
                        fifo_depth,
                        params_l1_ty,
                    )

                # Link: distribute packed params across rows
                if n_aie_rows > 1:
                    dist_offsets = [
                        row * params_per_channel for row in range(n_aie_rows)
                    ]
                else:
                    dist_offsets = []
                object_fifo_link(
                    params_l3l2[col],
                    [params_l2l1[row][col] for row in range(n_aie_rows)],
                    [],
                    dist_offsets,
                )

                # --- B_C: broadcast from shim -> mem -> all cores in column ---

                BC_l3l2[col] = object_fifo(
                    f"BC_L3L2_{col}",
                    shim_tiles[col],
                    mem_tiles[col],
                    fifo_depth,
                    BC_ty,
                )

                BC_l2l1[col] = object_fifo(
                    f"BC_L2L1_{col}",
                    mem_tiles[col],
                    [core_tiles[row][col] for row in range(n_aie_rows)],
                    fifo_depth,
                    BC_ty,
                )

                object_fifo_link(BC_l3l2[col], BC_l2l1[col])

                # --- Output: join from individual cores -> mem -> shim ---

                for row in range(n_aie_rows):
                    out_l1l2[row][col] = object_fifo(
                        f"out_L1L2_{col}_{row}",
                        core_tiles[row][col],
                        mem_tiles[col],
                        fifo_depth,
                        out_l1_ty,
                    )

                out_l2l3[col] = object_fifo(
                    f"out_L2L3_{col}",
                    mem_tiles[col],
                    shim_tiles[col],
                    fifo_depth,
                    out_l2_ty,
                )

                if n_aie_rows > 1:
                    join_offsets = [
                        row * out_per_channel for row in range(n_aie_rows)
                    ]
                else:
                    join_offsets = []
                object_fifo_link(
                    [out_l1l2[row][col] for row in range(n_aie_rows)],
                    out_l2l3[col],
                    join_offsets,
                    [],
                )

            # ============================================================
            # State buffers (local on each compute tile)
            # ============================================================

            state_buffers = [
                [None] * n_aie_cols for _ in range(n_aie_rows)
            ]
            for row in range(n_aie_rows):
                for col in range(n_aie_cols):
                    state_buffers[row][col] = buffer(
                        tile=core_tiles[row][col],
                        datatype=state_ty,
                        name=f"state_{col}_{row}",
                        initial_value=np.zeros(cpt * d_state, dtype=dtype),
                    )

            # ============================================================
            # Compute cores
            # ============================================================

            for row in range(n_aie_rows):
                for col in range(n_aie_cols):

                    @core(
                        core_tiles[row][col], "kernels.a", stack_size=0x1000
                    )
                    def core_body():
                        for _ in range_(sys.maxsize):
                            # Acquire inputs and output buffer
                            elem_params = params_l2l1[row][col].acquire(
                                ObjectFifoPort.Consume, 1
                            )
                            elem_BC = BC_l2l1[col].acquire(
                                ObjectFifoPort.Consume, 1
                            )
                            elem_out = out_l1l2[row][col].acquire(
                                ObjectFifoPort.Produce, 1
                            )

                            # Zero state and run interleaved multi-channel scan
                            zero_func(state_buffers[row][col], cpt * d_state)
                            mamba_scan_func(
                                elem_params,
                                elem_BC,
                                state_buffers[row][col],
                                elem_out,
                                seq_len,
                                d_state,
                                cpt,
                            )

                            # Release buffers
                            params_l2l1[row][col].release(
                                ObjectFifoPort.Consume, 1
                            )
                            BC_l2l1[col].release(
                                ObjectFifoPort.Consume, 1
                            )
                            out_l1l2[row][col].release(
                                ObjectFifoPort.Produce, 1
                            )

            # ============================================================
            # Runtime sequence (host DMA control)
            # ============================================================

            @runtime_sequence(
                np.ndarray[
                    (n_cores * params_per_channel,), np.dtype[dtype]
                ],
                np.ndarray[(B_C_sz,), np.dtype[dtype]],
                np.ndarray[
                    (n_cores * out_per_channel,), np.dtype[dtype]
                ],
            )
            def sequence(all_params, B_C, all_out):
                for col in range(n_aie_cols):
                    # Send packed per-channel params for this column
                    params_offset = col * params_per_col
                    npu_dma_memcpy_nd(
                        metadata=params_l3l2[col],
                        bd_id=0,
                        mem=all_params,
                        offsets=[0, 0, 0, params_offset],
                        sizes=[1, 1, 1, params_per_col],
                        strides=[0, 0, 0, 1],
                    )

                    # Broadcast shared B_C to this column
                    npu_dma_memcpy_nd(
                        metadata=BC_l3l2[col],
                        bd_id=1,
                        mem=B_C,
                        offsets=[0, 0, 0, 0],
                        sizes=[1, 1, 1, B_C_sz],
                        strides=[0, 0, 0, 1],
                    )

                    # Receive packed output from this column
                    out_offset = col * out_per_col
                    npu_dma_memcpy_nd(
                        metadata=out_l2l3[col],
                        bd_id=2,
                        mem=all_out,
                        offsets=[0, 0, 0, out_offset],
                        sizes=[1, 1, 1, out_per_col],
                        strides=[0, 0, 0, 1],
                    )

                # Wait for all output DMAs to complete
                dma_wait(*out_l2l3)

    print(ctx.module)


if __name__ == "__main__":
    main()
