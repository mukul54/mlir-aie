//===- mm_svdq.cc - svdq_gemm v2: W4A8 GEMM with in-kernel scale epilogue -===//
//
// Implements the integer branch of SVDQuant W4A8 with real quantization
// semantics (v1 = mm_w4a8.cc computed raw integer codes only):
//
//   C_f32[M,N] += ( A_int8[M,k] . unpack(B_int4[k,N]) )_int32 * sigma[g, N]
//
// summed over k-tiles g by repeated kernel calls. The per-(k-tile, column)
// weight scales sigma travel INLINE in the B stream (TileFuse-style metadata
// placement, arXiv:2606.11357), so no extra fifos are needed. Per-token
// activation scales commute out of the K-sum and are applied by the host/iGPU.
//
// B tile layout as streamed (host pre-tiled, linear DMA, no dims_to_stream):
//   [ (k/8 * n/8) blocks of 8x8 int4 nibbles, i-major j-minor, 32 B each ]
//   [ n bf16 per-column scales for this k-tile ]
//   total: k*n/2 + 2*n bytes
//
// Within a block, nibbles are row-major 8x8 with N-adjacent pairs per byte
// (low nibble = even column), so one aie::unpack yields the int8 MMUL operand.
// Compute: 8x8x8 int8 MMUL (512 MACs/op) with int32 accumulation; epilogue
// converts to float, multiplies by the column scales, accumulates into C f32.
//
// C is zeroed once per output tile (zero_f32) and accumulated across the K/k
// calls in f32 — v1's int8 C round-tripping (mod-256 wraparound) is gone.
//
// NOTE: int4* pointer arithmetic on this target is BYTE-granular; nibble-block
// stride is size_B/2 = 32 (see whole_array_w4a8/README.md).
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#include "zero.cc"

#ifndef DIM_M
#define DIM_M 64
#endif
#ifndef DIM_K
#define DIM_K 64
#endif
#ifndef DIM_N
#define DIM_N 64
#endif

template <unsigned rowA, unsigned colA, unsigned colB>
static inline void matmul_svdq_2x2(const int8 *__restrict pA,
                                   const int8 *__restrict pBraw,
                                   float *__restrict pC) {
  constexpr unsigned r = 8;
  constexpr unsigned s = 8;
  constexpr unsigned t = 8;
  using MMUL = aie::mmul<r, s, t, int8, int8, accauto>;
  constexpr unsigned B_BLK = MMUL::size_B / 2; // bytes per 8x8 nibble block

  // Scales live after all nibble blocks: colA*colB blocks * 32 B each.
  const bfloat16 *__restrict pScale =
      reinterpret_cast<const bfloat16 *>(pBraw + colA * colB * B_BLK);
  const int4 *__restrict pB = reinterpret_cast<const int4 *>(pBraw);

  event0();

  for (unsigned z = 0; z < rowA; z += 2)
    chess_prepare_for_pipelining chess_loop_range(4, ) {

      float *__restrict pC1 = pC + (z * colB) * MMUL::size_C;
      float *__restrict pC2 = pC + ((z + 1) * colB) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
        chess_flatten_loop
#endif
        {
          const int8 *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
          const int8 *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
          // int4* arithmetic is byte-granular: stride between blocks = B_BLK.
          const int4 *__restrict pB1 = pB + (j)*B_BLK;
          const int4 *__restrict pB2 = pB + (j + 1) * B_BLK;

          aie::vector<int8, MMUL::size_A> A0;
          aie::vector<int8, MMUL::size_A> A1;
          aie::vector<int8, MMUL::size_B> B0;
          aie::vector<int8, MMUL::size_B> B1;

          MMUL C00;
          MMUL C01;
          MMUL C10;
          MMUL C11;

          for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
            chess_flatten_loop
#endif
            {
              A0 = aie::load_v<MMUL::size_A>(pA1);
              pA1 += MMUL::size_A;
              A1 = aie::load_v<MMUL::size_A>(pA2);
              pA2 += MMUL::size_A;
              B0 = aie::unpack(aie::load_v<MMUL::size_B>(pB1));
              pB1 += B_BLK * colB;
              B1 = aie::unpack(aie::load_v<MMUL::size_B>(pB2));
              pB2 += B_BLK * colB;

              if (i == 0) {
                C00.mul(A0, B0);
                C01.mul(A0, B1);
                C10.mul(A1, B0);
                C11.mul(A1, B1);
              } else {
                C00.mac(A0, B0);
                C01.mac(A0, B1);
                C10.mac(A1, B0);
                C11.mac(A1, B1);
              }
            }

          // Epilogue: Cf32 += float(acc_int32) * sigma[column]
          // sigma for block column j covers output columns j*t .. j*t+7;
          // replicate the 8 scales down the r rows (row-major r x t layout).
          aie::vector<bfloat16, t> s0 = aie::load_v<t>(pScale + j * t);
          aie::vector<bfloat16, t> s1 = aie::load_v<t>(pScale + (j + 1) * t);
          // bf16 -> f32 via accfloat accumulator (no direct to_float for bf16)
          aie::accum<accfloat, MMUL::size_C> s0acc(
              s0.template grow_replicate<MMUL::size_C>());
          aie::accum<accfloat, MMUL::size_C> s1acc(
              s1.template grow_replicate<MMUL::size_C>());
          aie::vector<float, MMUL::size_C> S0 = s0acc.template to_vector<float>();
          aie::vector<float, MMUL::size_C> S1 = s1acc.template to_vector<float>();

          auto upd = [&](float *__restrict pCx,
                         aie::vector<float, MMUL::size_C> Sx, MMUL &Cx) {
            aie::vector<float, MMUL::size_C> acc_f =
                aie::to_float<float>(Cx.template to_vector<int32>());
            aie::vector<float, MMUL::size_C> cur =
                aie::load_v<MMUL::size_C>(pCx);
            aie::store_v(pCx, aie::add(cur, aie::mul(acc_f, Sx).template to_vector<float>()));
          };

          upd(pC1, S0, C00);
          pC1 += MMUL::size_C;
          upd(pC1, S1, C01);
          pC1 += MMUL::size_C;
          upd(pC2, S0, C10);
          pC2 += MMUL::size_C;
          upd(pC2, S1, C11);
          pC2 += MMUL::size_C;
        }
    }

  event1();
}

extern "C" {

void zero_f32(float *c_out) { zero_vectorized<float, DIM_M, DIM_N>(c_out); }

// A: (DIM_M x DIM_K) int8 codes.
// B: k*n/2 packed int4 blocks + n bf16 scales (see header comment).
// C: (DIM_M x DIM_N) f32, accumulated across calls.
void matmul_svdq(const int8 *a_in, const int8 *b_in, float *c_out) {
  static_assert(DIM_M % (2 * 8) == 0);
  static_assert(DIM_K % 8 == 0);
  static_assert(DIM_N % (2 * 8) == 0);
  matmul_svdq_2x2<(DIM_M / 8), (DIM_K / 8), (DIM_N / 8)>(a_in, b_in, c_out);
}

} // extern "C"
