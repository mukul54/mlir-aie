//===- mamba_scan.cc ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Mamba Selective Scan kernel for AIE2P
// Implements the core SSM state update: state = state * exp(dt * A) + B * x * dt
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

#include "../aie_kernel_utils.h"
#include "lut_based_ops.h"

#define REL_WRITE 0
#define REL_READ 1
#define LOG2E 1.4453125f  // log2(e) for exp(x) = 2^(log2e * x)

//==============================================================================
// Note: F32 scalar version removed - AIE2P exp only works with vector sizes >= 16
// Only BF16 vectorized version is provided (see mamba_scan_vector_bf16 below)
//==============================================================================

//==============================================================================
// Mamba Selective Scan - Vectorized Implementation (BFloat16)
//==============================================================================
//
// For d_state = 16, we process all 16 state dimensions at once using AIE vectors.
//
//==============================================================================

void mamba_scan_vector_bf16(
    const bfloat16* restrict x,
    const bfloat16* restrict dt,
    const bfloat16* restrict A,
    const bfloat16* restrict B,
    const bfloat16* restrict C,
    const bfloat16 D,
    bfloat16* restrict state,
    bfloat16* restrict out,
    const int32_t seq_len,
    const int32_t d_state
) {
    event0();

    // DEBUG: Write marker to prove kernel is called
    out[0] = bfloat16(42.0f);

    // Currently only support d_state = 16 for vectorization
    if (d_state != 16) {
        out[1] = bfloat16(99.0f);  // DEBUG: wrong d_state
        event1();
        return;
    }

    constexpr int VEC_SIZE = 16;
    using Vec16bf = aie::vector<bfloat16, VEC_SIZE>;

    // Load state into vector register (single 16-element vector for d_state=16)
    Vec16bf state_vec = aie::load_v<VEC_SIZE>((bfloat16*)state);

    // Load A matrix
    Vec16bf A_vec = aie::load_v<VEC_SIZE>((bfloat16*)A);

    // Sequential scan (parallelization limited by recurrence)
    for (int32_t t = 0; t < seq_len; t++) {
        bfloat16 x_t = x[t];
        bfloat16 dt_t = dt[t];

        // Broadcast scalars to vectors
        Vec16bf x_vec = aie::broadcast<bfloat16, VEC_SIZE>(x_t);
        Vec16bf dt_vec = aie::broadcast<bfloat16, VEC_SIZE>(dt_t);

        // Load B[t, :] and C[t, :]
        Vec16bf B_vec = aie::load_v<VEC_SIZE>((bfloat16*)B + t * d_state);
        Vec16bf C_vec = aie::load_v<VEC_SIZE>((bfloat16*)C + t * d_state);

        // Compute dt * A element-wise
        Vec16bf dt_A = aie::mul(dt_vec, A_vec);

        // Compute exp(dt * A) using LUT-based getExpBf16
        v16accfloat exp_acc = getExpBf16(dt_A);
        Vec16bf exp_vec = to_v16bfloat16(exp_acc);

        // State update: state = state * exp(dt*A) + B * x * dt
        aie::accum<accfloat, VEC_SIZE> state_acc = aie::mul(state_vec, exp_vec);

        Vec16bf x_dt_vec = aie::mul(x_vec, dt_vec);
        Vec16bf B_x_dt = aie::mul(B_vec, x_dt_vec);

        state_acc = aie::add(state_acc, B_x_dt);
        state_vec = state_acc.to_vector<bfloat16>();

        // Output: y = sum(state * C)
        Vec16bf y_contrib = aie::mul(state_vec, C_vec);

        // Horizontal reduction to sum all elements
        bfloat16 y_sum = aie::reduce_add(y_contrib);

        // Add skip connection
        out[t] = y_sum + bfloat16(float(D) * float(x_t));
    }

    // Write back final state
    aie::store_v((bfloat16*)state, state_vec);

    event1();
}

//==============================================================================
// Wrapper Functions (exported to MLIR)
//==============================================================================

extern "C" {

// F32 scalar version not supported on AIE2P - use BF16 instead

// Vectorized bfloat16 version with combined buffers
// Buffer layout (to fit within NPU1 shim tile's 2 S2MM DMA channels):
//   params: [x[0..seq_len-1], dt[0..seq_len-1], A[0..d_state-1], D, padding]
//   B_C:    [B[flatten], C[flatten]]
void mamba_scan_bf16(
    bfloat16* params,  // Combined buffer: [x, dt, A, D, pad]
    bfloat16* B_C,     // Combined buffer: [B[flatten], C[flatten]]
    bfloat16* state,
    bfloat16* out,
    int32_t seq_len,
    int32_t d_state
) {
    // Extract individual arrays from combined buffers
    // params layout: [x: seq_len, dt: seq_len, A: d_state, D: 1, pad: 1]
    bfloat16* x = params;                           // x is at start
    bfloat16* dt = params + seq_len;                // dt is after x
    bfloat16* A = params + seq_len + seq_len;       // A is after dt
    bfloat16 D = params[seq_len + seq_len + d_state]; // D is after A

    bfloat16* B = B_C;                              // B is at start
    bfloat16* C = B_C + (seq_len * d_state);        // C is after B

    mamba_scan_vector_bf16(x, dt, A, B, C, D, state, out, seq_len, d_state);
}

// Zero initialization helper
void zero_f32(float* buffer, int32_t size) {
    for (int32_t i = 0; i < size; i++) {
        buffer[i] = 0.0f;
    }
}

void zero_bf16(bfloat16* buffer, int32_t size) {
    for (int32_t i = 0; i < size; i++) {
        buffer[i] = bfloat16(0.0f);
    }
}

} // extern "C"
