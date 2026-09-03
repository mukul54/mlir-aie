# whole_array_w4a8 — W4A8 GEMM (int4-packed weights, int8 MMUL)

A variant of `whole_array` where the B (weight) matrix is stored and moved as
**packed int4** and widened to int8 inside the cores, in front of the standard
8x8x8 int8 MMUL. A (activations) and C (output) stay int8. This is the integer
branch of an SVDQuant-style W4A8 deployment: 4-bit weight storage, 8-bit
integer compute, 2x less B DMA than int8 and 4x less than bf16.

Measured on a Ryzen AI 9 HX 370 (Strix, npu2, 8 columns), M=1024 K=1600 N=4096,
m=k=n=64, all three kernels built from this tree, results bit-exact vs an
int32 reference:

| kernel               | avg latency | TOPS | vs bf16 |
|----------------------|-------------|------|---------|
| whole_array bf16     | 8976 us     | 1.50 | 1.0x    |
| whole_array i8       | 2091 us     | 6.42 | 4.3x    |
| **whole_array_w4a8** | **1645 us** | **8.16** | **5.5x** |

## Packing convention (host side)

B pairs adjacent **N** columns into one byte, row-major, low nibble first:

    B_packed[r, c] = (B[r, 2c] & 0xF) | (B[r, 2c+1] << 4)   # shape (K, N/2), int8
    values are signed int4 in [-8, 7]

Packing along N (not K) is what makes the in-core unpack trivial: after the
L2->L1 stream reorder below, each 32-byte block is exactly one row-major
s x t int4 MMUL operand, so a single `aie::unpack` of a `vector<int4,64>`
yields the `vector<int8,64>` the MMUL consumes. No shuffles needed.

## What changed vs `whole_array/whole_array.py`

All B-side extents are halved (`nb = n//2`, `Nb = N//2`, `tb4 = t//2`);
A and C paths are untouched:

1. `B_l2_ty` / `B_l1_ty`: `(k * n)` -> `(k * nb)` bytes.
2. L2->L1 `dims_to_stream` for B: `[(k//s, s*n), (n//t, t), (s, n), (t, 1)]`
   -> `[(k//s, s*nb), (nb//tb4, tb4), (s, nb), (tb4, 1)]`
   (blocks of s x t/2 *bytes* = s x t nibbles, row-major).
3. Shim DMA for B: offset `col*n` -> `col*nb`;
   sizes `[N//n//cols, K//k, k, n]` -> `[..., k, nb]`;
   strides `[n*cols, k*N, N, 1]` -> `[nb*cols, k*Nb, Nb, 1]`.
4. Runtime sequence B tensor type: `(K*N,)` -> `(K*Nb,)`.
5. Kernel: external func `matmul_w4a8` from object `mm_w4a8_{m}x{k}x{n}.o`
   (source: `aie_kernels/aie2p/mm_w4a8.cc`). `zero_i8` comes from the same
   object (it includes `mm.cc` with `-Di8_i8_ONLY`).
6. Row-major B only (`b_col_maj` asserted off).

## The kernel (`aie_kernels/aie2p/mm_w4a8.cc`)

Copy of the 2x2-unrolled 8x8x8 int8 MMUL loop with the B loads replaced by

    B0 = aie::unpack(aie::load_v<64>(pB1));   // int4 -> int8 widen in-core

**Pitfall that cost a debug round:** `int4*` pointer arithmetic on this target
is *byte-granular*, not nibble-granular. The stride between consecutive
64-nibble B blocks is therefore `MMUL::size_B / 2` (= 32), not `size_B`.
Getting this wrong silently skips every other block (diagnose with an
identity probe: set A[:,0]=1 so each C row returns B row 0 verbatim).

## Build

    make devicename=npu2 M=1024 K=1600 N=4096 m=64 k=64 n=64 n_aie_cols=8 \
         dtype_in=i8 dtype_out=i8

Constraints: use the default (IRON) `whole_array_w4a8.py` path; m, k, n and
column counts follow the same divisibility rules as `whole_array`, plus
n and t even. Tested with the mlir_aie wheels dated 2026-02.
