#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.
import argparse        # Command-line argument parsing
import numpy as np     # For dtype definitions
import sys             # For exit codes

from aie.extras.context import mlir_mod_ctx        # MLIR context manager
from aie.dialects.aie import *                     # AIE dialect (tile, objectfifo, etc.)
from aie.dialects.aiex import *                    # AIEx dialect (runtime ops)
import aie.utils.trace as trace_utils              # Performance tracing
from aie.utils.trace import PortEvent              # Trace event definitions
from aie.helpers.dialects.ext.scf import _for as range_  # Loop constructs
from aie.iron.dtype import str_to_dtype            # String to numpy dtype converter



microkernel_mac_dim_map = {
    "npu": {                    # NPU1 (Ryzen AI)
        "bf16": (4, 8, 4),      # r=4, s=8, t=4
        "i8": (4, 8, 8),
        "i16": (4, 4, 4),       # Default for this example
    },
    "npu2": {                   # NPU2 (newer generation)
        "bf16": {
            True: (8, 8, 8),    # Emulated mode
            False: (4, 8, 8),   # Native mode
        },
        "i8": (8, 8, 8),
        "i16": (4, 4, 8),
    },
}



def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Single Core)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("-M", type=int, default=256)  # Full matrix rows
    argparser.add_argument("-K", type=int, default=256)  # Inner dimension
    argparser.add_argument("-N", type=int, default=256)  # Full matrix cols
    argparser.add_argument("-m", type=int, default=64)   # Tile rows
    argparser.add_argument("-k", type=int, default=64)   # Tile inner dim
    argparser.add_argument("-n", type=int, default=32)   # Tile cols
    argparser.add_argument(
        "--dtype_in", type=str, choices=["bf16", "i8", "i16"], default="i16"
    )
    argparser.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "f32", "i32"],
        default="i32",
    )
    argparser.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--emulate-bf16-mmul-with-bfp16", type=bool, default=False)
    argparser.add_argument("--trace_size", type=int, default=0)
    args = argparser.parse_args()
    my_matmul(
        args.dev,
        args.M,
        args.K,
        args.N,
        args.m,
        args.k,
        args.n,
        args.dtype_in,
        args.dtype_out,
        args.b_col_maj,
        args.emulate_bf16_mmul_with_bfp16,
        args.trace_size,
    )


def ceildiv(a, b):
    return (a + b - 1) // b


def my_matmul(
    dev,
    M,
    K,
    N,
    m,
    k,
    n,
    dtype_in_str,
    dtype_out_str,
    b_col_maj,
    emulate_bf16_mmul_with_bfp16,
    trace_size,
):

    assert M % m == 0
    assert K % k == 0
    assert N % n == 0

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    mac_dims = microkernel_mac_dim_map[dev][dtype_in_str]
    if dev == "npu2" and dtype_in_str == "bf16":
        r, s, t = mac_dims[emulate_bf16_mmul_with_bfp16]
    else:
        r, s, t = mac_dims

    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    vectorized = True
    enable_tracing = True if trace_size > 0 else False

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    A_sz = M * K
    B_sz = K * N
    C_sz = M * N

    M_div_m = M // m # 256/64 = 4
    K_div_k = K // k # 256/64 = 4
    N_div_n = N // n # 256/32 = 8
    tiles = M_div_m * N_div_n # 4*8 = 32

    with mlir_mod_ctx() as ctx:

        if dev == "npu":
            dev_ty = AIEDevice.npu1_1col # Target Ryzen AI (Phoenix)
        else:
            dev_ty = AIEDevice.npu2 # Target newer NPU

        # @device(dev_ty) decorator starts MLIR generation and tells the compiler what hardware to target
        @device(dev_ty)
        def device_body():
            # Everything inside device_body() will be translated to MLIR AIE dialect
            a_ty = np.ndarray[(m, k), np.dtype[dtype_in]]      # a[64, 64] of int16
            b_ty = np.ndarray[(k, n), np.dtype[dtype_in]]      # b[64, 32] of int16  
            c_ty = np.ndarray[(m, n), np.dtype[dtype_out]]     # c[64, 32] of int32

            # AIE Core Function declarations
            # Declare external kernel functions (implemented in C++)
            func_type = "" if vectorized else "scalar_"
            zero = external_func(f"zero_{func_type}{dtype_out_str}", inputs=[c_ty])
            # Declares: zero_i32(c[64, 32]), zero_i32 → Implemented in zero.cc

            matmul_func_name = f"matmul_{func_type}{dtype_in_str}_{dtype_out_str}"
            matmul = external_func(
                matmul_func_name,
                inputs=[a_ty, b_ty, c_ty],
            )
            # Declares: matmul_i16_i32(a[64, 64], b[64, 32], c[64, 32]), matmul_i16_i32 → Implemented in mm.cc

            # Tile declarations
            shim_tile = tile(0, 0) # Column 0, Row 0 (Interface to DRAM)
            mem_tile = tile(0, 1) # Column 0, Row 1 (L2 cache)
            compute_tile2_col, compute_tile2_row = 0, 2
            compute_tile2 = tile(compute_tile2_col, compute_tile2_row) # Column 0, Row 2 (Compute core)

            """
            Visual Layout:
            Row 2: [Compute Tile] ← Runs matmul kernel
                    ↕
            Row 1: [  Mem Tile  ] ← L2 buffer, data transformation
                    ↕
            Row 0: [ Shim Tile  ] ← Interface to DRAM
                    ↕
                [DRAM (L3)]
            """
            # AIE-array data movement with object fifos
            # Input A
            # L3 → L2: Transfer 64x64 tiles
            inA = object_fifo("inA", shim_tile, mem_tile, 2, a_ty)
            #                  name   producer  consumer depth type

            """
            "inA" - Name for debugging/tracing
            shim_tile - Producer (reads from DRAM)
            mem_tile - Consumer (receives data)
            2 - Depth (double buffering - 2 buffers)
            a_ty - Type: int16[64, 64]

            meaning of depth 2::

            Time 0: Buffer 0 filling from DRAM, Buffer 1 being consumed
            Time 1: Buffer 1 filling from DRAM, Buffer 0 being consumed
                    ↑ Overlap communication & computation!
            """
            # L2 → L1: Rearrange data for vectorization
            memA = object_fifo(
                "memA",
                mem_tile, # Producer
                compute_tile2, # Consumer
                2,  # Depth
                a_ty, # Type: int16[64, 64]
                (
                    [ # ← TRANSFORMATION PATTERN
                        (m // r, r * k), # (16, 256) - 16 groups of 256 elements
                        (k // s, s), # (16, 4)   - Each group has 16 sub-groups of 4
                        (r, k), # (4, 64)   - Row stride
                        (s, 1), # (4, 1)    - Element stride
                    ]
                    if vectorized
                    else []
                ),
            )

            """
            Understanding the Transformation: Original layout in memory (row-major A[64, 64]):
            Row 0: [a00, a01, a02, a03, a04, ..., a63]
            Row 1: [a10, a11, a12, a13, a14, ..., a63]
            ...
            After transformation (grouped for aie::mmul<4, 4, 4>):

            Group 0 (MMUL tile 0):
            [a00, a01, a02, a03,  ← First 4 elements
            a10, a11, a12, a13,  ← of first 4 rows
            a20, a21, a22, a23,
            a30, a31, a32, a33]  ← Total: 4x4 = 16 elements

            Group 1 (MMUL tile 1):
            [a04, a05, a06, a07,  ← Next 4 elements
            a14, a15, a16, a17,  ← of first 4 rows
            ...
            Why? The hardware mmul intrinsic expects data in this grouped format for efficient SIMD loading.
            """
            object_fifo_link(inA, memA) # Connect the two FIFOs
            """
            DRAM → [inA: Shim→Mem] → [memA: Mem→Compute] → Compute Core
                    (no transform)        (with transform)

            """
            # Input B
            inB = object_fifo("inB", shim_tile, mem_tile, 2, b_ty)

            B_transformations = []
            # Transformation depends on whether B is row-major or column-major
            if vectorized:
                if not b_col_maj:
                    B_transformations = [
                        (k // s, s * n), # (16, 128) - 16 K-groups
                        (n // t, t), # (8, 4)    - 8 N-groups of 4
                        (s, n), # (4, 32)   - K-stride
                        (t, 1), # (4, 1)    - Element stride
                    ]
                else:
                    B_transformations = [
                        (n // t, t * k),
                        (k // s, s),
                        (t, k),
                        (s, 1),
                    ]

            """
            For row-major B[64, 32]: The transformation reorganizes B to match the mmul access pattern:
            Original B (row-major):
            k0: [b00, b01, ..., b031]  ← 32 columns
            k1: [b10, b11, ..., b131]
            ...

            Transformed (for MMUL):
            Group 0: [b00, b01, b02, b03,  ← First 4 cols
                        b10, b11, b12, b13,  ← of first 4 rows
                        b20, b21, b22, b23,
                        b30, b31, b32, b33]  ← 4x4 tile
            """

            memB = object_fifo(
                "memB",
                mem_tile,
                compute_tile2,
                2,
                b_ty,
                B_transformations,
            )

            object_fifo_link(inB, memB)

            # Output C
            # L1 → L2: Compute produces output
            memC = object_fifo("memC", compute_tile2, mem_tile, 2, c_ty)
            outC = object_fifo(
                "outC",
                mem_tile,
                shim_tile, # Consumer (writes to DRAM)
                2,
                c_ty,
                (
                    [
                        (m // r, r * n),
                        (r, t),
                        (n // t, r * t),
                        (t, 1),
                    ]
                    if vectorized
                    else []
                ),
            )
            object_fifo_link(memC, outC)
            # Compute Core → [memC: vectorized layout] → [outC: row-major] → DRAM

            # Set up a packet-switched flow from core to shim for tracing information
            tiles_to_trace = [compute_tile2]
            if trace_size > 0:
                trace_utils.configure_packet_tracing_flow(tiles_to_trace, shim_tile)

            # The stack size choice is an important choice!
            # The Peano compiler uses a stack size in this kernel greater than the default one
            # (default is 0x400, chess' stack size is smaller).
            # Exceding the stack size leads to wrong results from the kernel, but no error is triggered.
            # Stack usage can be checked as explained here:
            # https://github.com/Xilinx/llvm-aie/issues/487#issuecomment-2969438585


            # compute_tile2 - Which tile runs this code
            # f"mm_64x64x32.o" - Compiled kernel object file
            # stack_size=0xD00 - 3328 bytes stack (default 0x400 is too small!)
            @core(compute_tile2, f"mm_{m}x{k}x{n}.o", stack_size=0xD00)
            def core_body():
                for _ in range_(0xFFFFFFFF): # Infinite loop (wait for tasks)
                    # Process all output tiles: For M=256, N=256, m=64, n=32:
                    # tiles = (256/64) * (256/32) = 4 * 8 = 32 output tiles
                    for _ in range_(tiles) if tiles > 1 else range(1):  # issue #1547 # # 32 iterations

                        elem_out = memC.acquire(ObjectFifoPort.Produce, 1)
                        zero(elem_out) # Initialize output tile to zeros
                        """
                        acquire() ────┐
                                      │ ← Work on buffer
                        zero()        │
                        matmul()      │
                                      │
                        release() ────┘

                        Why zero first? Because we accumulate across K dimension:
                        C = A[:, 0:64] @ B[0:64, :] + 
                            A[:, 64:128] @ B[64:128, :] + 
                            A[:, 128:192] @ B[128:192, :] + 
                            A[:, 192:256] @ B[192:256, :]

                        """
                        for _ in (
                            range_(K_div_k) if K_div_k > 1 else range(1)
                        ):  # issue #1547 # # 4 iterations
                            elem_in_a = memA.acquire(ObjectFifoPort.Consume, 1)
                            elem_in_b = memB.acquire(ObjectFifoPort.Consume, 1)

                            ############ Kernel operation ####
                            matmul(elem_in_a, elem_in_b, elem_out) # C += A @ B
                            ################################################

                            memA.release(ObjectFifoPort.Consume, 1)
                            memB.release(ObjectFifoPort.Consume, 1)
                            """
                            This implements the K-loop:
                            for k_tile in [0, 1, 2, 3]:
                                a = A[:, k_tile*64:(k_tile+1)*64]    # Get 64×64 tile of A
                                b = B[k_tile*64:(k_tile+1)*64, :]    # Get 64×32 tile of B
                                C += a @ b                            # Accumulate

                            """
                        memC.release(ObjectFifoPort.Produce, 1) # Output tile complete

            # To/from AIE-array data movement

            # This is the HOST-SIDE control program that runs on the CPU and sends data to the NPU.
            @runtime_sequence(
                np.ndarray[(A_sz,), np.dtype[dtype_in]],
                np.ndarray[(B_sz,), np.dtype[dtype_in]],
                np.ndarray[(C_sz,), np.dtype[dtype_out]],
            )
            def sequence(A, B, C):

                if enable_tracing:
                    trace_utils.configure_packet_tracing_aie2(
                        tiles_to_trace=tiles_to_trace,
                        shim=shim_tile,
                        trace_size=trace_size,
                        coretile_events=[
                            # captures input A (PORT_RUNNING_0, at port number 1, master for inputs)
                            trace_utils.PortEvent(
                                trace_utils.CoreEvent.PORT_RUNNING_0,
                                port_number=1,
                                master=True,
                            ),
                            # captures input B (PORT_RUNNING_1, at port number 2, master for inputs)
                            trace_utils.PortEvent(
                                trace_utils.CoreEvent.PORT_RUNNING_1,
                                port_number=2,
                                master=True,
                            ),
                            # captures output C (PORT_RUNNING_2, at port number 1, slave for outputs)
                            trace_utils.PortEvent(
                                trace_utils.CoreEvent.PORT_RUNNING_2,
                                port_number=1,
                                master=False,
                            ),
                            trace_utils.CoreEvent.INSTR_EVENT_0,
                            trace_utils.CoreEvent.INSTR_EVENT_1,
                            trace_utils.CoreEvent.MEMORY_STALL,
                            trace_utils.CoreEvent.LOCK_STALL,
                            trace_utils.CoreEvent.INSTR_VECTOR,
                        ],
                    )

                # only do 4 tile rows at a time before synchronizing, so we can reuse BDs (Buffer Descriptors)
                rows_per_block = 4 # Process 4 tile rows at a time
                for tile_row_block in range(ceildiv(M_div_m, rows_per_block)):
                    # we only sync on half the BDs before reusing them, so the other half can concurrently keep running
                    # that's what this loop is for
                    for pingpong in [0, 1]: # Ping-pong buffering
                        C_row_offset = (
                            tile_row_block * rows_per_block * m * N
                            + pingpong * rows_per_block // 2 * m * N
                        )
                        row_base = (
                            tile_row_block * rows_per_block
                            + pingpong * rows_per_block // 2
                        )
                        bd_id_base = 8 * pingpong # BDs 0-7 or 8-15
                        num_tile_rows = min([rows_per_block // 2, M_div_m - row_base])
                        if num_tile_rows <= 0:
                            # At the very last iteration, we may not need a 'pong' iteration
                            break

                        """
                        Buffer Descriptor (BD) Management: Shim tiles have only 16 BDs (hardware resources). We split them:
                        Ping Buffer Descriptors: BDs 0-7   ← Processing block 0
                        Pong Buffer Descriptors: BDs 8-15  ← Processing block 1
                        Ping-pong execution:

                        Time 0: BDs 0-7 transferring data, BDs 8-15 waiting
                        Time 1: BDs 8-15 transferring data, BDs 0-7 reused (after dma_wait)
                        This allows overlapping data transfers!
                        """
                        npu_dma_memcpy_nd(
                            metadata=outC,                      # Which ObjectFIFO
                            bd_id=bd_id_base,                   # Buffer descriptor ID
                            mem=C,                              # DRAM buffer
                            offsets=[0, 0, 0, C_row_offset],   # 4D offset
                            sizes=[num_tile_rows, N // n, m, n], # [2, 8, 64, 32]
                            strides=[m * N, n, N, 1],           # [16384, 32, 256, 1]
                        )
                        """
                        Understanding 4D DMA: This is a nested loop in hardware:

                        // Equivalent pseudo-code
                        for (d0 = 0; d0 < sizes[0]; d0++) {         // num_tile_rows (e.g., 2)
                        for (d1 = 0; d1 < sizes[1]; d1++) {       // N // n (8)
                            for (d2 = 0; d2 < sizes[2]; d2++) {     // m (64)
                            for (d3 = 0; d3 < sizes[3]; d3++) {   // n (32)
                                addr = offsets[3] + 
                                    d0 * strides[0] +             // Jump to next tile row
                                    d1 * strides[1] +             // Jump to next tile col
                                    d2 * strides[2] +             // Jump to next row in tile
                                    d3 * strides[3];              // Next element
                                
                                transfer_element(C[addr]);
                            }
                            }
                        }
                        }
                        Visualized for output C:

                        sizes = [2, 8, 64, 32]
                                │  │  │   └── 32 elements per row (innermost)
                                │  │  └────── 64 rows per tile
                                │  └───────── 8 tiles across (N // n)
                                └──────────── 2 tile rows this iteration

                        strides = [16384, 32, 256, 1]
                                    │      │    │  └── Next element: offset += 1
                                    │      │    └──────── Next row: offset += 256 (N)
                                    │      └──────────── Next tile col: offset += 32
                                    └─────────────────── Next tile row: offset += 64*256
                        """

                        for tile_row in range(num_tile_rows):
                            A_row_offset = (row_base + tile_row) * m * K # Start of this tile row
                            npu_dma_memcpy_nd(
                                metadata=inA,
                                bd_id=bd_id_base + 2 * tile_row + 1,
                                mem=A,
                                offsets=[0, 0, 0, A_row_offset],
                                sizes=[N // n, K // k, m, k], # [8, 4, 64, 64]
                                strides=[0, k, K, 1], # [0, 64, 256, 1]
                            )

                            """
                            Key insight - strides[0] = 0: This reuses the same A tile row across all N output tiles!
                            For output tile (row=0, col=0): A[0:64, :]
                            For output tile (row=0, col=1): A[0:64, :]  ← Same A!
                            For output tile (row=0, col=2): A[0:64, :]  ← Reused
                            ...
                            Why? In matrix multiplication, the same row of A multiplies with different columns of B:

                            C[0:64, 0:32]   = A[0:64, :] @ B[:, 0:32]
                            C[0:64, 32:64]  = A[0:64, :] @ B[:, 32:64]  ← Same A reused!
                            C[0:64, 64:96]  = A[0:64, :] @ B[:, 64:96]  ← Same A reused!
                            """

                            if not b_col_maj:  # Row-major B
                                B_sizes = [N // n, K // k, k, n]       # [8, 4, 64, 32]
                                B_strides = [n, k * N, N, 1]           # [32, 16384, 256, 1]
                            else:  # Column-major B
                                B_sizes = [N // n, K // k, n, k]
                                B_strides = [n * K, k, K, 1]


                            """
                            For row-major B:

                            strides = [32, 16384, 256, 1]
                                        │   │      │    └── Next element
                                        │   │      └──────── Next row within K-tile
                                        │   └─────────────── Next K-tile (jump 64 rows)
                                        └─────────────────── Next N-tile (jump 32 cols)
                            This accesses B in tiles:

                            Iteration 0: B[0:64, 0:32]     (top-left)
                            Iteration 1: B[0:64, 32:64]    (next column tile)
                            Iteration 2: B[64:128, 0:32]   (next K-tile)
                            ...

                            """
                            npu_dma_memcpy_nd(
                                metadata=inB,
                                bd_id=bd_id_base + 2 * tile_row + 2,
                                mem=B,
                                sizes=B_sizes,
                                strides=B_strides,
                            )
                        if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                            dma_wait(outC) # Wait for previous block to complete

                dma_wait(outC) # Final wait
                """
                Why wait? Before reusing BDs, we must ensure previous transfers are done:

                Block 0, Ping:  Use BDs 0-7
                                ↓ (start transfer)
                Block 0, Pong:  Use BDs 8-15
                                ↓ (start transfer)
                                dma_wait()  ← Wait for Ping before reusing BDs 0-7
                Block 1, Ping:  Reuse BDs 0-7 (safe now!)
                """

    print(ctx.module)


if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)
