//===- mm_w4a8.cc - W4A8 GEMM microkernel for AIE2P (svdq_gemm v1) -------===//
//
// Integer branch of the SVDQuant dual-branch co-design:
//   A: int8 (m x k), streamed in r x s blocks (identical to the i8 kernel)
//   B: int4-PACKED along N pairs. Streamed as s x (t/2)-byte blocks, i.e.
//      s x t nibbles, little-endian nibble order == row-major s x t int4 block.
//      Each MMUL B-operand (64 nibbles = 32 bytes) is loaded as
//      aie::vector<int4,64> and widened to int8 in-core (aie::unpack).
//   C: int8 (m x n)
//   MMUL: 8x8x8 int8 (512 MACs/op) — same compute path as the validated i8
//   kernel; B DMA traffic is HALVED vs int8, 4x vs bf16.
//
// Compile with -Di8_i8_ONLY -DDIM_M=.. -DDIM_K=.. -DDIM_N=.. (the include
// below also provides zero_i8 and the reference matmul_i8_i8 symbols).
//
//===----------------------------------------------------------------------===//

#include "mm.cc"

template <unsigned rowA, unsigned colA, unsigned colB>
static inline void matmul_vectorized_2x2_mmul_w4(const int8 *__restrict pA,
                                                 const int4 *__restrict pB,
                                                 int8 *__restrict pC) {
  constexpr unsigned r = 8;
  constexpr unsigned s = 8;
  constexpr unsigned t = 8;
  using MMUL = aie::mmul<r, s, t, int8, int8, accauto>;

  event0();

  for (unsigned z = 0; z < rowA; z += 2)
    chess_prepare_for_pipelining chess_loop_range(4, ) {

      int8 *__restrict pC1 = pC + (z * colB) * MMUL::size_C;
      int8 *__restrict pC2 = pC + ((z + 1) * colB) * MMUL::size_C;

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

          aie::vector<int8, MMUL::size_C> acc_C00 =
              aie::load_v<MMUL::size_C>(pC1);
          aie::vector<int8, MMUL::size_C> acc_C01 =
              aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
          aie::vector<int8, MMUL::size_C> acc_C10 =
              aie::load_v<MMUL::size_C>(pC2);
          aie::vector<int8, MMUL::size_C> acc_C11 =
              aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);

          MMUL C00(acc_C00);
          MMUL C01(acc_C01);
          MMUL C10(acc_C10);
          MMUL C11(acc_C11);

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

              C00.mac(A0, B0);
              C01.mac(A0, B1);
              C10.mac(A1, B0);
              C11.mac(A1, B1);
            }

          aie::store_v(pC1, C00.template to_vector<int8>());
          pC1 += MMUL::size_C;
          aie::store_v(pC1, C01.template to_vector<int8>());
          pC1 += MMUL::size_C;
          aie::store_v(pC2, C10.template to_vector<int8>());
          pC2 += MMUL::size_C;
          aie::store_v(pC2, C11.template to_vector<int8>());
          pC2 += MMUL::size_C;
        }
    }

  event1();
}

extern "C" {

// A: (DIM_M x DIM_K) int8, B: packed int4 (DIM_K x DIM_N/2 bytes), C: int8.
void matmul_w4a8(const int8 *a_in, const int8 *b_packed_in, int8 *c_out) {
  static_assert(DIM_M % (2 * 8) == 0);
  static_assert(DIM_K % 8 == 0);
  static_assert(DIM_N % (2 * 8) == 0);
  matmul_vectorized_2x2_mmul_w4<(DIM_M / 8), (DIM_K / 8), (DIM_N / 8)>(
      a_in, reinterpret_cast<const int4 *>(b_packed_in), c_out);
}

} // extern "C"
