//===- mm_w4sh.cc - W4A8 GEMM, int32 C, FREE per-group pow2 scales (svdq v4) -===//
//
//   C_i32[M,N] += ( A_i8[M,k] . unpack(B_i4[k,N]) )_i32  >> sh[ktile, N]
//
// Weight quantization: W[n,k] ~= s_n * 2^-sh[g,n] * q, q int4. The per-channel
// float s_n (and the per-token activation scale) are applied OUTSIDE the kernel
// on the iGPU. The per-(k-tile, column) integer shift sh is applied here for
// free: it rides the accumulator readout (srs) that the store needs anyway,
// with rounding mode = floor (== arithmetic right shift). The kernel requires
// sh to be constant within each 8-column MMUL block (host guarantees).
//
// B tile stream layout per k-tile (host pre-tiled, linear):
//   [ (k/8 x n/8) 8x8 int4 nibble blocks, i-major j-minor ][ n int8 shifts ]
//
// Compile with -Di8_i32_ONLY (mm.cc then also provides zero_i32).
//
//===----------------------------------------------------------------------===//

#include "mm.cc"

template <unsigned rowA, unsigned colA, unsigned colB>
static inline void matmul_w4sh_2x2(const int8 *__restrict pA,
                                   const int8 *__restrict pBbytes,
                                   int32 *__restrict pC) {
  constexpr unsigned r = 8;
  constexpr unsigned s = 8;
  constexpr unsigned t = 8;
  using MMUL = aie::mmul<r, s, t, int8, int8, accauto>;
  const int4 *__restrict pB = reinterpret_cast<const int4 *>(pBbytes);
  // per-column shifts follow the nibble blocks: (k * n / 2) bytes in
  const int8 *__restrict pSh = pBbytes + (colA * s) * (colB * t) / 2;
  aie::set_rounding(aie::rounding_mode::floor);   // srs shift == floor(acc / 2^sh)

  event0();

  for (unsigned z = 0; z < rowA; z += 2)
    chess_prepare_for_pipelining chess_loop_range(4, ) {

      int32 *__restrict pC1 = pC + (z * colB) * MMUL::size_C;
      int32 *__restrict pC2 = pC + ((z + 1) * colB) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
        chess_flatten_loop
#endif
        {
          const int8 *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
          const int8 *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
          // NOTE: int4* pointer arithmetic is BYTE-granular on this target
          // (verified empirically: +64 advanced 64 bytes = 2 blocks). One
          // 64-nibble MMUL block = 32 bytes, so stride in "+" units is
          // MMUL::size_B / 2.
          constexpr unsigned B_BLK = MMUL::size_B / 2; // bytes per B block
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
              // int4 -> int8 widen in-core; layout is already the row-major
              // s x t block the MMUL expects.
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

          // Epilogue: C_i32 += acc >> sh[block column]  (shift rides the srs)
          const int sh0 = pSh[j * t];
          const int sh1 = pSh[(j + 1) * t];
          auto upd = [&](int32 *__restrict pCx, int shx, MMUL &Cx) {
            aie::vector<int32, MMUL::size_C> cur = aie::load_v<MMUL::size_C>(pCx);
            aie::store_v(pCx, aie::add(cur, Cx.template to_vector<int32>(shx)));
          };
          upd(pC1, sh0, C00);
          pC1 += MMUL::size_C;
          upd(pC1, sh1, C01);
          pC1 += MMUL::size_C;
          upd(pC2, sh0, C10);
          pC2 += MMUL::size_C;
          upd(pC2, sh1, C11);
          pC2 += MMUL::size_C;
        }
    }

  event1();
}

extern "C" {

// A: (DIM_M x DIM_K) int8. B: DIM_K*DIM_N/2 packed int4 bytes + DIM_N int8 shifts.
// C: (DIM_M x DIM_N) int32, accumulated across calls.
void matmul_w4sh(const int8 *a_in, const int8 *b_in, int32 *c_out) {
  static_assert(DIM_M % (2 * 8) == 0);
  static_assert(DIM_K % 8 == 0);
  static_assert(DIM_N % (2 * 8) == 0);
  matmul_w4sh_2x2<(DIM_M / 8), (DIM_K / 8), (DIM_N / 8)>(a_in, b_in, c_out);
}

} // extern "C"
