//===- conv1dk_i8.cc --------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#define REL_WRITE 0
#define REL_READ 1

const int32_t SMAX = 127;
const int32_t SMIN = -128;

#ifdef SCALAR

//*****************************************************************************
// Conv1D - Scalar Implementation (Standard Padding)
//
// Input:  [length, in_channels]
// Weights: [kernel_size, in_channels, out_channels]
// Output: [length, out_channels]
//
// Padding: Asymmetric (works for both odd and even kernel sizes)
//   - For kernel_size=3: pad_left=1, pad_right=1 (symmetric)
//   - For kernel_size=4: pad_left=1, pad_right=2 (asymmetric)
//
// Memory Layout:
//   Input[t, ic]  = input[t * in_channels + ic]
//   Weight[k, ic, oc] = kernels[(k * in_channels * out_channels) + 
//                               (ic * out_channels) + oc]
//   Output[t, oc] = output[t * out_channels + oc]
//
// For efficiency, channels are processed in groups of 8.
//*****************************************************************************
void conv1dk_i8_scalar(int8_t *input, int8_t *kernels, int8_t *output,
                       const int32_t length, const int32_t in_channels,
                       const int32_t out_channels, const int32_t kernel_size,
                       const int32_t scale) {
  event0();

  const int32_t pad_left = (kernel_size - 1) / 2;

  for (int oc8_idx = 0; oc8_idx < out_channels / 8; oc8_idx++) {
    for (int t = 0; t < length; t++) {
      for (int oc_sub = 0; oc_sub < 8; oc_sub++) {
        int oc = oc8_idx * 8 + oc_sub;
        int32_t accumulator = 0;

        for (int k = 0; k < kernel_size; k++) {
          int32_t t_in = t + k - pad_left;

          if (t_in >= 0 && t_in < length) {
            for (int ic8_idx = 0; ic8_idx < in_channels / 8; ic8_idx++) {
              for (int ic_sub = 0; ic_sub < 8; ic_sub++) {
                int ic = ic8_idx * 8 + ic_sub;

                int8_t input_val = input[t_in * in_channels + ic];
                
                int kernel_idx = (k * in_channels * out_channels) +
                                 (ic8_idx * out_channels * 8) +
                                 (oc8_idx * 64) +
                                 (ic_sub * 8) +
                                 oc_sub;
                int8_t kernel_val = kernels[kernel_idx];

                accumulator += (int32_t)input_val * (int32_t)kernel_val;
              }
            }
          }
        }

        int32_t scaled;
        // 1. Check for the bad case
        if (scale == 0) {
          scaled = accumulator; // No scaling, just pass through
        } 
        // 2. Run the normal logic if scale is valid
        else {
          // This is the rounding + re-scaling
          scaled = (accumulator + (1 << (scale - 1))) >> scale; 
        }

        // 3. Apply clamping in all cases
        scaled = (scaled > SMAX) ? SMAX : (scaled < SMIN) ? SMIN : scaled;
        output[t * out_channels + oc] = (int8_t)scaled;
      }
    }
  }

  event1();
}

//*****************************************************************************
// Conv1D - Scalar Implementation (Causal Padding)
//
// Used for autoregressive models like Mamba where future information
// must not be used. The convolution at position 't' only depends on
// positions [t - kernel_size + 1, t].
//
// Padding: Only on the left (past)
//   - For kernel_size=4: pad_left=3, pad_right=0
//   - Output at position t depends only on inputs at positions t-3, t-2, t-1, t
//
// This ensures causality for sequential generation tasks.
//*****************************************************************************
void conv1dk_i8_causal_scalar(int8_t *input, int8_t *kernels, int8_t *output,
                              const int32_t length, const int32_t in_channels,
                              const int32_t out_channels, const int32_t kernel_size,
                              const int32_t scale) {
  event0();

  const int32_t pad_left = kernel_size - 1;

  for (int oc8_idx = 0; oc8_idx < out_channels / 8; oc8_idx++) {
    for (int t = 0; t < length; t++) {
      for (int oc_sub = 0; oc_sub < 8; oc_sub++) {
        int oc = oc8_idx * 8 + oc_sub;
        int32_t accumulator = 0;

        for (int k = 0; k < kernel_size; k++) {
          int32_t t_in = t + k - pad_left;

          if (t_in >= 0) {
            for (int ic8_idx = 0; ic8_idx < in_channels / 8; ic8_idx++) {
              for (int ic_sub = 0; ic_sub < 8; ic_sub++) {
                int ic = ic8_idx * 8 + ic_sub;

                int8_t input_val = input[t_in * in_channels + ic];
                
                int kernel_idx = (k * in_channels * out_channels) +
                                 (ic8_idx * out_channels * 8) +
                                 (oc8_idx * 64) +
                                 (ic_sub * 8) +
                                 oc_sub;
                int8_t kernel_val = kernels[kernel_idx];

                accumulator += (int32_t)input_val * (int32_t)kernel_val;
              }
            }
          }
        }

        int32_t scaled;
        // 1. Check for the bad case
        if (scale == 0) {
          scaled = accumulator; // No scaling, just pass through
        } 
        // 2. Run the normal logic if scale is valid
        else {
          // This is the rounding + re-scaling
          scaled = (accumulator + (1 << (scale - 1))) >> scale; 
        }
        // 3. Apply clamping in all cases
        scaled = (scaled > SMAX) ? SMAX : (scaled < SMIN) ? SMIN : scaled;
        output[t * out_channels + oc] = (int8_t)scaled;
      }
    }
  }

  event1();
}

#else // VECTOR

//*****************************************************************************
// Conv1D - Vectorized Implementation (Standard Padding)
//
// Uses AIE vector intrinsics for high performance:
//   - Processes 4 time positions simultaneously
//   - Uses 8 parallel accumulators
//   - Vectorized loads/stores (32 elements at a time)
//
// Processing strategy:
//   1. Main loop: Process chunks of 32 positions (4*8)
//   2. Remainder loop: Handle remaining positions (if length % 32 != 0)
//
// Each MMUL operation processes:
//   - 4 time positions
//   - 8 input channels
//   - 8 output channels
//   - Result: 4x8 = 32 output values
//
// Channels must be multiples of 8 for vector processing.
//*****************************************************************************
void conv1dk_i8_vector(int8_t *input, int8_t *kernels, int8_t *output,
                       const int32_t length, const int32_t in_channels,
                       const int32_t out_channels, const int32_t kernel_size,
                       const int32_t scale) {
  event0();

  constexpr int NUM_ACC = 8;
  constexpr int MMUL_M = 4;
  constexpr int MMUL_K = 8;
  constexpr int MMUL_N = 8;
  constexpr int CHANNEL_FACTOR = MMUL_K;
  constexpr int MMUL_MK = MMUL_M * MMUL_K;
  constexpr int MMUL_KN = MMUL_K * MMUL_N;
  constexpr int MMUL_MN = MMUL_M * MMUL_N;

  using MMUL4x8x8 = aie::mmul<MMUL_M, MMUL_K, MMUL_N, int8, int8>;

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::symmetric_inf);

  int8_t *restrict out_ptr = output;
  const int scaleT = scale;

  MMUL4x8x8 acc_tmp[NUM_ACC];
  for (int i = 0; i < NUM_ACC; i++) {
    acc_tmp[i] = aie::zeros<acc32, MMUL_MN>();
  }

  const int32_t pad_left = (kernel_size - 1) / 2;
  const int len = length;
  const int len_partial = (length / MMUL_M) / NUM_ACC;
  const int len_partial_rem = (length / MMUL_M) % NUM_ACC;

  int8_t *input_begin = input;
  int8_t *input_rem_begin = input + len_partial * MMUL_M * NUM_ACC * CHANNEL_FACTOR;

  //==========================================================================
  // Main Loop: Process chunks of 32 positions
  //==========================================================================
  if (len_partial > 0) {
    for (int oc_grp = 0; oc_grp < (out_channels / CHANNEL_FACTOR); oc_grp++) {
      for (int len_partial_idx = 0; len_partial_idx < len_partial; len_partial_idx++) {
        
        for (int k = 0; k < kernel_size; k++) {
          int32_t k_offset = k - pad_left;
          
          AIE_PREPARE_FOR_PIPELINING
          AIE_LOOP_MIN_ITERATION_COUNT(2)
          for (int ic_grp = 0; ic_grp < (in_channels / CHANNEL_FACTOR); ic_grp++) {
            aie::vector<int8, MMUL_KN> wts = aie::load_v<MMUL_KN>(kernels);
            kernels += MMUL_KN;

            for (int acc_idx = 0; acc_idx < NUM_ACC; acc_idx++) {
              int32_t pos = len_partial_idx * NUM_ACC * MMUL_M + acc_idx * MMUL_M;
              int32_t input_pos = pos + k_offset;

              if (input_pos >= 0 && input_pos + MMUL_M - 1 < len) {
                aie::vector<int8, MMUL_MK> acts = aie::load_v<MMUL_MK>(input);
                input += MMUL_MK;
                acc_tmp[acc_idx].mac(acts, wts);
              } else {
                input += MMUL_MK;
              }
            }
            input += (len * CHANNEL_FACTOR) - MMUL_MK * NUM_ACC;
          }
          
          input -= (in_channels * len) - MMUL_MK * NUM_ACC;
          kernels -= (in_channels / CHANNEL_FACTOR) * MMUL_KN;
        }

        for (int acc_idx = 0; acc_idx < NUM_ACC; acc_idx++) {
          aie::vector<int8, MMUL_MN> result = acc_tmp[acc_idx].to_vector<int8>(scaleT);
          aie::store_v(out_ptr, result);
          out_ptr += MMUL_MN;
          acc_tmp[acc_idx] = aie::zeros<acc32, MMUL_MN>();
        }
        
        input -= (in_channels * len) - MMUL_MK * NUM_ACC;
        kernels -= (in_channels / CHANNEL_FACTOR) * MMUL_KN * kernel_size;
      }
      
      input = input_begin;
      kernels += (in_channels / CHANNEL_FACTOR) * MMUL_KN * kernel_size;
      out_ptr += (len_partial_rem * MMUL_MN);
    }
  }

  //==========================================================================
  // Remainder Loop: Handle positions that don't fit in chunks of 32
  //==========================================================================
  if (len_partial_rem > 0) {
    const int ocs = out_channels;
    const int ics = in_channels;

    for (int oc_grp = 0; oc_grp < (ocs / CHANNEL_FACTOR); oc_grp++) {
      
      for (int k = 0; k < kernel_size; k++) {
        int32_t k_offset = k - pad_left;
        
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_MIN_ITERATION_COUNT(2)
        for (int ic_grp = 0; ic_grp < (ics / CHANNEL_FACTOR); ic_grp++) {
          aie::vector<int8, MMUL_KN> wts = aie::load_v<MMUL_KN>(kernels);
          kernels += MMUL_KN;

          for (int acc_idx = 0; acc_idx < len_partial_rem; acc_idx++) {
            int32_t pos = len_partial * NUM_ACC * MMUL_M + acc_idx * MMUL_M;
            int32_t input_pos = pos + k_offset;

            if (input_pos >= 0 && input_pos + MMUL_M - 1 < len) {
              aie::vector<int8, MMUL_MK> acts = aie::load_v<MMUL_MK>(input);
              input += MMUL_MK;
              acc_tmp[acc_idx].mac(acts, wts);
            } else {
              input += MMUL_MK;
            }
          }
          input += (len * CHANNEL_FACTOR) - (MMUL_MK * len_partial_rem);
        }
        
        input = input_rem_begin;
        kernels -= (ics / CHANNEL_FACTOR) * MMUL_KN;
      }

      for (int acc_idx = 0; acc_idx < len_partial_rem; acc_idx++) {
        aie::vector<int8, MMUL_MN> result = acc_tmp[acc_idx].to_vector<int8>(scaleT);
        aie::store_v(out_ptr, result);
        out_ptr += MMUL_MN;
        acc_tmp[acc_idx] = aie::zeros<acc32, MMUL_MN>();
      }

      input = input_rem_begin;
      kernels += (ics / CHANNEL_FACTOR) * MMUL_KN * kernel_size;
      out_ptr += (len * CHANNEL_FACTOR) - (len_partial_rem * MMUL_MN);
    }
  }

  event1();
}

//*****************************************************************************
// Conv1D - Vectorized Implementation (Causal Padding)
//
// Same as standard version but with causal padding (only look backward).
// Critical for autoregressive models where future information must not leak.
//
// Padding: pad_left = kernel_size - 1, pad_right = 0
//
// At position t, the convolution sees inputs from [t-kernel_size+1, t].
// This maintains causality for sequential generation.
//*****************************************************************************
void conv1dk_i8_causal_vector(int8_t *input, int8_t *kernels, int8_t *output,
                              const int32_t length, const int32_t in_channels,
                              const int32_t out_channels, const int32_t kernel_size,
                              const int32_t scale) {
  event0();

  constexpr int NUM_ACC = 8;
  constexpr int MMUL_M = 4;
  constexpr int MMUL_K = 8;
  constexpr int MMUL_N = 8;
  constexpr int CHANNEL_FACTOR = MMUL_K;
  constexpr int MMUL_MK = MMUL_M * MMUL_K;
  constexpr int MMUL_KN = MMUL_K * MMUL_N;
  constexpr int MMUL_MN = MMUL_M * MMUL_N;

  using MMUL4x8x8 = aie::mmul<MMUL_M, MMUL_K, MMUL_N, int8, int8>;

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::symmetric_inf);

  int8_t *restrict out_ptr = output;
  const int scaleT = scale;

  MMUL4x8x8 acc_tmp[NUM_ACC];
  for (int i = 0; i < NUM_ACC; i++) {
    acc_tmp[i] = aie::zeros<acc32, MMUL_MN>();
  }

  const int32_t pad_left = kernel_size - 1;
  const int len = length;
  const int len_partial = (length / MMUL_M) / NUM_ACC;
  const int len_partial_rem = (length / MMUL_M) % NUM_ACC;

  int8_t *input_begin = input;
  int8_t *input_rem_begin = input + len_partial * MMUL_M * NUM_ACC * CHANNEL_FACTOR;

  if (len_partial > 0) {
    for (int oc_grp = 0; oc_grp < (out_channels / CHANNEL_FACTOR); oc_grp++) {
      for (int len_partial_idx = 0; len_partial_idx < len_partial; len_partial_idx++) {
        
        for (int k = 0; k < kernel_size; k++) {
          int32_t k_offset = k - pad_left;
          
          AIE_PREPARE_FOR_PIPELINING
          AIE_LOOP_MIN_ITERATION_COUNT(2)
          for (int ic_grp = 0; ic_grp < (in_channels / CHANNEL_FACTOR); ic_grp++) {
            aie::vector<int8, MMUL_KN> wts = aie::load_v<MMUL_KN>(kernels);
            kernels += MMUL_KN;

            for (int acc_idx = 0; acc_idx < NUM_ACC; acc_idx++) {
              int32_t pos = len_partial_idx * NUM_ACC * MMUL_M + acc_idx * MMUL_M;
              int32_t input_pos = pos + k_offset;

              if (input_pos >= 0) {
                aie::vector<int8, MMUL_MK> acts = aie::load_v<MMUL_MK>(input);
                input += MMUL_MK;
                acc_tmp[acc_idx].mac(acts, wts);
              } else {
                input += MMUL_MK;
              }
            }
            input += (len * CHANNEL_FACTOR) - MMUL_MK * NUM_ACC;
          }
          
          input -= (in_channels * len) - MMUL_MK * NUM_ACC;
          kernels -= (in_channels / CHANNEL_FACTOR) * MMUL_KN;
        }

        for (int acc_idx = 0; acc_idx < NUM_ACC; acc_idx++) {
          aie::vector<int8, MMUL_MN> result = acc_tmp[acc_idx].to_vector<int8>(scaleT);
          aie::store_v(out_ptr, result);
          out_ptr += MMUL_MN;
          acc_tmp[acc_idx] = aie::zeros<acc32, MMUL_MN>();
        }
        
        input -= (in_channels * len) - MMUL_MK * NUM_ACC;
        kernels -= (in_channels / CHANNEL_FACTOR) * MMUL_KN * kernel_size;
      }
      
      input = input_begin;
      kernels += (in_channels / CHANNEL_FACTOR) * MMUL_KN * kernel_size;
      out_ptr += (len_partial_rem * MMUL_MN);
    }
  }

  if (len_partial_rem > 0) {
    const int ocs = out_channels;
    const int ics = in_channels;

    for (int oc_grp = 0; oc_grp < (ocs / CHANNEL_FACTOR); oc_grp++) {
      
      for (int k = 0; k < kernel_size; k++) {
        int32_t k_offset = k - pad_left;
        
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_MIN_ITERATION_COUNT(2)
        for (int ic_grp = 0; ic_grp < (ics / CHANNEL_FACTOR); ic_grp++) {
          aie::vector<int8, MMUL_KN> wts = aie::load_v<MMUL_KN>(kernels);
          kernels += MMUL_KN;

          for (int acc_idx = 0; acc_idx < len_partial_rem; acc_idx++) {
            int32_t pos = len_partial * NUM_ACC * MMUL_M + acc_idx * MMUL_M;
            int32_t input_pos = pos + k_offset;

            if (input_pos >= 0) {
              aie::vector<int8, MMUL_MK> acts = aie::load_v<MMUL_MK>(input);
              input += MMUL_MK;
              acc_tmp[acc_idx].mac(acts, wts);
            } else {
              input += MMUL_MK;
            }
          }
          input += (len * CHANNEL_FACTOR) - (MMUL_MK * len_partial_rem);
        }
        
        input = input_rem_begin;
        kernels -= (ics / CHANNEL_FACTOR) * MMUL_KN;
      }

      for (int acc_idx = 0; acc_idx < len_partial_rem; acc_idx++) {
        aie::vector<int8, MMUL_MN> result = acc_tmp[acc_idx].to_vector<int8>(scaleT);
        aie::store_v(out_ptr, result);
        out_ptr += MMUL_MN;
        acc_tmp[acc_idx] = aie::zeros<acc32, MMUL_MN>();
      }

      input = input_rem_begin;
      kernels += (ics / CHANNEL_FACTOR) * MMUL_KN * kernel_size;
      out_ptr += (len * CHANNEL_FACTOR) - (len_partial_rem * MMUL_MN);
    }
  }

  event1();
}

#endif // VECTOR

//*****************************************************************************
// Wrapper Functions
//*****************************************************************************
extern "C" {

void conv1dk_i8(int8_t *input, int8_t *kernels, int8_t *output,
                const int32_t length, const int32_t in_channels,
                const int32_t out_channels, const int32_t kernel_size,
                const int32_t scale) {
#ifdef SCALAR
  conv1dk_i8_scalar(input, kernels, output, length, in_channels,
                    out_channels, kernel_size, scale);
#else
  conv1dk_i8_vector(input, kernels, output, length, in_channels,
                    out_channels, kernel_size, scale);
#endif
}

void conv1dk_i8_causal(int8_t *input, int8_t *kernels, int8_t *output,
                       const int32_t length, const int32_t in_channels,
                       const int32_t out_channels, const int32_t kernel_size,
                       const int32_t scale) {
#ifdef SCALAR
  conv1dk_i8_causal_scalar(input, kernels, output, length, in_channels,
                           out_channels, kernel_size, scale);
#else
  conv1dk_i8_causal_vector(input, kernels, output, length, in_channels,
                           out_channels, kernel_size, scale);
#endif
}

} // extern "C"