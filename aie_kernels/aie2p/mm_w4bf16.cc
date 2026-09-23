//===- mm_w4bf16.cc - svdq_gemm v5: W4A8 GEMM, bf16 output, ONE epilogue per tile ===//
//
//   C_bf16[M,N] = bf16( ( A_int8[M,K] . unpack(B_int4[K,N]) )_int32 * s[N] )
//
// The design calls this kernel once per k-tile for each output tile (K_TILES
// calls between two zero_bf16 calls). The int32 accumulators live in a STATIC
// per-core buffer across those calls; only the LAST call converts them to
// float, applies the per-output-channel weight scale s (inline in the B
// stream, same layout as svdq v2) and stores bf16 straight into the C tile.
// So, unlike v2 (epilogue every k-tile) and v3 (int32 out + GPU finalize),
// the NPU writes the final bf16 output and the iGPU does no post-work; the
// per-token activation scale is folded into the consumer (GeGLU) on the iGPU.
//
// B tile stream (host pre-tiled, linear): [ k*n/2 nibble bytes ][ n bf16 scales ]
// Compile with -Di8_bf16_ONLY -DDIM_M -DDIM_K -DDIM_N -DK_TILES=K/DIM_K.
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

#ifndef K_TILES
#define K_TILES 16
#endif

// int32 accumulators for one DIM_M x DIM_N output tile, persistent across the
// K_TILES calls of one tile (block-major order, same as the C tile). Zeroed by
// zero_bf16 at tile start so every call can preload + mac (no branch in the loops).
// One static object only: [0] = k-tile call counter, [64..] = accumulators. Keeping
// them in a single array with a 256-byte gap avoids the two separate .bss objects
// landing on each other in core memory (seen as acc[0] += counter on aie2p).
alignas(32) static int32 w4_state[64 + DIM_M * DIM_N];
#define k_call (w4_state[0])
#define acc_static (w4_state + 64)

template <bool LAST, unsigned rowA, unsigned colA, unsigned colB>
static inline void matmul_w4bf16_body(const int8 *__restrict pA,
                                      const int8 *__restrict pBraw,
                                      bfloat16 *__restrict pC) {
  constexpr unsigned r = 8;
  constexpr unsigned s = 8;
  constexpr unsigned t = 8;
  using MMUL = aie::mmul<r, s, t, int8, int8, accauto>;
  constexpr unsigned B_BLK = MMUL::size_B / 2; // bytes per 8x8 nibble block

  const bfloat16 *__restrict pScale =
      reinterpret_cast<const bfloat16 *>(pBraw + colA * colB * B_BLK);
  const int4 *__restrict pB = reinterpret_cast<const int4 *>(pBraw);

  for (unsigned z = 0; z < rowA; z += 2)
    chess_prepare_for_pipelining chess_loop_range(4, ) {

      int32 *__restrict pAcc1 = acc_static + (z * colB) * MMUL::size_C;
      int32 *__restrict pAcc2 = acc_static + ((z + 1) * colB) * MMUL::size_C;
      bfloat16 *__restrict pC1 = pC + (z * colB) * MMUL::size_C;
      bfloat16 *__restrict pC2 = pC + ((z + 1) * colB) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
        chess_flatten_loop
#endif
        {
          const int8 *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
          const int8 *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
          const int4 *__restrict pB1 = pB + (j)*B_BLK;
          const int4 *__restrict pB2 = pB + (j + 1) * B_BLK;

          aie::vector<int8, MMUL::size_A> A0;
          aie::vector<int8, MMUL::size_A> A1;
          aie::vector<int8, MMUL::size_B> B0;
          aie::vector<int8, MMUL::size_B> B1;

          // preload the running int32 accumulators (exactly like v3 w4i32)
          MMUL C00(aie::load_v<MMUL::size_C>(pAcc1));
          MMUL C01(aie::load_v<MMUL::size_C>(pAcc1 + MMUL::size_C));
          MMUL C10(aie::load_v<MMUL::size_C>(pAcc2));
          MMUL C11(aie::load_v<MMUL::size_C>(pAcc2 + MMUL::size_C));

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

              C00.mac(A0, B0);
              C01.mac(A0, B1);
              C10.mac(A1, B0);
              C11.mac(A1, B1);
            }

          if constexpr (!LAST) {
            aie::store_v(pAcc1, C00.template to_vector<int32>());
            aie::store_v(pAcc1 + MMUL::size_C, C01.template to_vector<int32>());
            aie::store_v(pAcc2, C10.template to_vector<int32>());
            aie::store_v(pAcc2 + MMUL::size_C, C11.template to_vector<int32>());
          } else {
            // Epilogue (once per tile): C_bf16 = bf16(float(acc) * s[column])
            aie::vector<bfloat16, t> s0 = aie::load_v<t>(pScale + j * t);
            aie::vector<bfloat16, t> s1 = aie::load_v<t>(pScale + (j + 1) * t);
            aie::accum<accfloat, MMUL::size_C> s0acc(s0.template grow_replicate<MMUL::size_C>());
            aie::accum<accfloat, MMUL::size_C> s1acc(s1.template grow_replicate<MMUL::size_C>());
            aie::vector<float, MMUL::size_C> S0 = s0acc.template to_vector<float>();
            aie::vector<float, MMUL::size_C> S1 = s1acc.template to_vector<float>();
            aie::store_v(pC1, aie::mul(aie::to_float<float>(C00.template to_vector<int32>()), S0)
                                  .template to_vector<bfloat16>());
            aie::store_v(pC1 + MMUL::size_C,
                         aie::mul(aie::to_float<float>(C01.template to_vector<int32>()), S1)
                             .template to_vector<bfloat16>());
            aie::store_v(pC2, aie::mul(aie::to_float<float>(C10.template to_vector<int32>()), S0)
                                  .template to_vector<bfloat16>());
            aie::store_v(pC2 + MMUL::size_C,
                         aie::mul(aie::to_float<float>(C11.template to_vector<int32>()), S1)
                             .template to_vector<bfloat16>());
          }
          pAcc1 += 2 * MMUL::size_C;
          pAcc2 += 2 * MMUL::size_C;
          pC1 += 2 * MMUL::size_C;
          pC2 += 2 * MMUL::size_C;
        }
    }
}

template <unsigned rowA, unsigned colA, unsigned colB>
static inline void matmul_w4bf16_2x2(const int8 *__restrict pA,
                                     const int8 *__restrict pBraw,
                                     bfloat16 *__restrict pC) {
  event0();
  if ((unsigned)k_call + 1 >= (unsigned)K_TILES) {
    matmul_w4bf16_body<true, rowA, colA, colB>(pA, pBraw, pC);
    k_call = 0;
  } else {
    matmul_w4bf16_body<false, rowA, colA, colB>(pA, pBraw, pC);
    ++k_call;
  }
  event1();
}

extern "C" {

// Called by the design at the start of every output tile: also restarts the k-tile count.
void zero_bf16(bfloat16 *c_out) {
  zero_vectorized<int32, DIM_M, DIM_N>(acc_static);
  k_call = 0;
  zero_vectorized<bfloat16, DIM_M, DIM_N>(c_out);
}

// A: (DIM_M x DIM_K) int8 codes. B: k*n/2 packed int4 blocks + n bf16 scales.
// C: (DIM_M x DIM_N) bf16, written on the last of K_TILES calls.
void matmul_w4bf16(const int8 *a_in, const int8 *b_in, bfloat16 *c_out) {
  static_assert(DIM_M % (2 * 8) == 0);
  static_assert(DIM_K % 8 == 0);
  static_assert(DIM_N % (2 * 8) == 0);
  matmul_w4bf16_2x2<(DIM_M / 8), (DIM_K / 8), (DIM_N / 8)>(a_in, b_in, c_out);
}

} // extern "C"
