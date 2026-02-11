//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Test harness for Mamba Selective Scan on AIE
//
//===----------------------------------------------------------------------===//

#include <bits/stdc++.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdfloat>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// Datatypes
#ifndef DTYPE
#define DTYPE std::bfloat16_t
#endif
using DATATYPE = DTYPE;

// ============================================================================
// CPU Reference Implementation
// ============================================================================

void mamba_scan_cpu_reference(
    const float* x,
    const float* dt,
    const float* A,
    const float* B,
    const float* C,
    const float D,
    float* state,
    float* out,
    int seq_len,
    int d_state
) {
    // Initialize state to zero
    for (int n = 0; n < d_state; n++) {
        state[n] = 0.0f;
    }

    // Sequential scan
    for (int t = 0; t < seq_len; t++) {
        float x_t = x[t];
        float dt_t = dt[t];
        float y_sum = 0.0f;

        for (int n = 0; n < d_state; n++) {
            float B_t = B[t * d_state + n];
            float C_t = C[t * d_state + n];

            // State update
            float exp_dt_A = expf(dt_t * A[n]);
            state[n] = state[n] * exp_dt_A + B_t * x_t * dt_t;

            // Output projection
            y_sum += state[n] * C_t;
        }

        // Skip connection
        out[t] = y_sum + D * x_t;
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

template <typename T>
T get_random() {
    if constexpr (std::is_same_v<T, float>) {
        return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    } else if constexpr (std::is_same_v<T, std::bfloat16_t>) {
        float val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
        return std::bfloat16_t(val);
    }
    return T(0);
}

template <typename T>
float to_float(T val) {
    if constexpr (std::is_same_v<T, float>) {
        return val;
    } else if constexpr (std::is_same_v<T, std::bfloat16_t>) {
        return static_cast<float>(val);
    }
    return 0.0f;
}

bool verify_results(const float* cpu_out, const float* npu_out, int seq_len,
                   float abs_tol = 0.5f, float rel_tol = 0.05f) {
    // Note: BFloat16 has limited precision (7-bit mantissa vs 23 for float32)
    // These tolerances match those used in matrix_multiplication for bf16
    int errors = 0;
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;

    for (int i = 0; i < seq_len; i++) {
        float abs_err = std::abs(cpu_out[i] - npu_out[i]);
        float rel_err = abs_err / (std::abs(cpu_out[i]) + 1e-8f);

        max_abs_error = std::max(max_abs_error, abs_err);
        max_rel_error = std::max(max_rel_error, rel_err);

        if (abs_err > abs_tol && rel_err > rel_tol) {
            if (errors < 10) {  // Print first 10 errors
                std::cerr << "Error at index " << i << ": "
                         << "CPU=" << cpu_out[i] << " NPU=" << npu_out[i]
                         << " abs_err=" << abs_err << " rel_err=" << rel_err << "\n";
            }
            errors++;
        }
    }

    std::cout << "Max absolute error: " << max_abs_error << "\n";
    std::cout << "Max relative error: " << max_rel_error << "\n";
    std::cout << "Total errors: " << errors << " / " << seq_len << "\n";

    return errors == 0;
}

std::vector<uint32_t> load_instr_binary(const std::string& instr_path) {
    std::ifstream instr_file(instr_path, std::ios::binary | std::ios::ate);
    if (!instr_file.is_open()) {
        throw std::runtime_error("Failed to open instruction file: " + instr_path);
    }

    std::streamsize size = instr_file.tellg();
    instr_file.seekg(0, std::ios::beg);

    std::vector<uint32_t> instr_v(size / sizeof(uint32_t));
    if (!instr_file.read(reinterpret_cast<char*>(instr_v.data()), size)) {
        throw std::runtime_error("Failed to read instruction file");
    }

    return instr_v;
}

// ============================================================================
// Main Test
// ============================================================================

int main(int argc, const char *argv[]) {
    // Parse arguments
    std::string xclbin_path = "final.xclbin";
    std::string instr_path = "insts.txt";
    std::string kernel_name = "MLIR_AIE";
    int seq_len = 128;
    int d_state = 16;
    bool do_verify = true;
    int n_iterations = 1;
    int verbosity = 1;

    // Simple argument parsing
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--xclbin" && i + 1 < argc) {
            xclbin_path = argv[++i];
        } else if (arg == "--instr" && i + 1 < argc) {
            instr_path = argv[++i];
        } else if (arg == "--kernel" && i + 1 < argc) {
            kernel_name = argv[++i];
        } else if (arg == "--seq_len" && i + 1 < argc) {
            seq_len = std::atoi(argv[++i]);
        } else if (arg == "--d_state" && i + 1 < argc) {
            d_state = std::atoi(argv[++i]);
        } else if (arg == "--verify") {
            do_verify = std::atoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            n_iterations = std::atoi(argv[++i]);
        } else if (arg == "-v" && i + 1 < argc) {
            verbosity = std::atoi(argv[++i]);
        }
    }

    if (verbosity >= 1) {
        std::cout << "====================================\n";
        std::cout << "Mamba Selective Scan Test\n";
        std::cout << "====================================\n";
        std::cout << "Sequence length: " << seq_len << "\n";
        std::cout << "State dimension: " << d_state << "\n";
        std::cout << "====================================\n";
    }

    // Fix random seed for reproducibility
    srand(42);

    // Calculate buffer sizes - matching the combined layout in mamba_scan.py
    // Combined buffer 1: [x, dt, A, D, padding]
    int x_sz = seq_len;
    int dt_sz = seq_len;
    int A_sz = d_state;
    int D_sz = 2;  // D + padding for 32-bit alignment
    int params_sz = x_sz + dt_sz + A_sz + D_sz;  // Combined params buffer

    // Combined buffer 2: [B flatten, C flatten]
    int B_sz = seq_len * d_state;
    int C_sz = seq_len * d_state;
    int B_C_sz = B_sz + C_sz;

    int out_sz = seq_len;

    // Load instruction sequence (binary format)
    std::vector<uint32_t> instr_v;
    try {
        instr_v = load_instr_binary(instr_path);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    if (verbosity >= 1) {
        std::cout << "Loaded " << instr_v.size() << " instructions\n";
    }

    // Initialize XRT
    auto device = xrt::device(0);
    auto xclbin = xrt::xclbin(xclbin_path);

    if (verbosity >= 1) {
        std::cout << "Loading xclbin: " << xclbin_path << "\n";
    }

    device.register_xclbin(xclbin);
    xrt::hw_context context(device, xclbin.get_uuid());

    auto xkernels = xclbin.get_kernels();
    auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                                 [kernel_name](xrt::xclbin::kernel &k) {
                                     return k.get_name().rfind(kernel_name, 0) == 0;
                                 });
    auto kernel = xrt::kernel(context, xkernel.get_name());

    // Allocate buffers matching the runtime_sequence signature:
    // sequence(Params, B_C, Out)
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                           XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_params = xrt::bo(device, params_sz * sizeof(DATATYPE),
                            XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_B_C = xrt::bo(device, B_C_sz * sizeof(DATATYPE),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_out = xrt::bo(device, out_sz * sizeof(DATATYPE),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    // Generate test data
    if (verbosity >= 1) {
        std::cout << "Generating test data...\n";
    }

    DATATYPE *buf_params = bo_params.map<DATATYPE*>();
    DATATYPE *buf_B_C = bo_B_C.map<DATATYPE*>();
    DATATYPE *buf_out = bo_out.map<DATATYPE*>();

    // CPU-side storage for verification
    std::vector<float> x_cpu(x_sz), dt_cpu(dt_sz), A_cpu(A_sz);
    std::vector<float> B_cpu(B_sz), C_cpu(C_sz);
    float D_cpu;

    // Initialize inputs and pack into combined buffers
    // Params layout: [x[0..seq_len-1], dt[0..seq_len-1], A[0..d_state-1], D, padding]
    int offset = 0;

    // x values
    for (int i = 0; i < x_sz; i++) {
        x_cpu[i] = get_random<float>() * 0.5f;  // Scale down for numerical stability
        buf_params[offset + i] = DATATYPE(x_cpu[i]);
    }
    offset += x_sz;

    // dt values
    for (int i = 0; i < dt_sz; i++) {
        dt_cpu[i] = std::abs(get_random<float>()) * 0.1f + 0.01f;  // Small positive values
        buf_params[offset + i] = DATATYPE(dt_cpu[i]);
    }
    offset += dt_sz;

    // A values
    for (int i = 0; i < A_sz; i++) {
        A_cpu[i] = -std::abs(get_random<float>()) * 0.5f;  // Negative for stability
        buf_params[offset + i] = DATATYPE(A_cpu[i]);
    }
    offset += A_sz;

    // D value and padding
    D_cpu = get_random<float>() * 0.5f;
    buf_params[offset] = DATATYPE(D_cpu);
    buf_params[offset + 1] = DATATYPE(0.0f);  // padding

    // B_C layout: [B[flatten], C[flatten]]
    offset = 0;

    // B values
    for (int i = 0; i < B_sz; i++) {
        B_cpu[i] = get_random<float>() * 0.3f;
        buf_B_C[offset + i] = DATATYPE(B_cpu[i]);
    }
    offset += B_sz;

    // C values
    for (int i = 0; i < C_sz; i++) {
        C_cpu[i] = get_random<float>() * 0.3f;
        buf_B_C[offset + i] = DATATYPE(C_cpu[i]);
    }

    memset(buf_out, 0, out_sz * sizeof(DATATYPE));

    // Write instruction buffer
    void *buf_instr = bo_instr.map<void*>();
    memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

    // Sync buffers to device
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_params.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_B_C.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Run kernel
    if (verbosity >= 1) {
        std::cout << "Running kernel...\n";
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < n_iterations; iter++) {
        unsigned int opcode = 3;  // Transaction opcode
        auto run = kernel(opcode, bo_instr, instr_v.size(),
                         bo_params, bo_B_C, bo_out);
        run.wait();
    }

    auto end = std::chrono::high_resolution_clock::now();
    double kernel_time_ms = std::chrono::duration<double, std::milli>(end - start).count() / n_iterations;

    // Sync output back
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (verbosity >= 1) {
        std::cout << "Kernel time: " << kernel_time_ms << " ms\n";
        // Calculate theoretical GOPS (2 ops per element: mul + add in state update)
        double gops = (2.0 * seq_len * d_state) / (kernel_time_ms * 1e6);
        std::cout << "Throughput: " << gops << " GOPS\n";
    }

    // Verification
    bool pass = true;
    if (do_verify) {
        if (verbosity >= 1) {
            std::cout << "Running CPU reference...\n";
        }

        std::vector<float> state_cpu(d_state);
        std::vector<float> out_cpu(out_sz);

        mamba_scan_cpu_reference(
            x_cpu.data(), dt_cpu.data(), A_cpu.data(),
            B_cpu.data(), C_cpu.data(), D_cpu,
            state_cpu.data(), out_cpu.data(),
            seq_len, d_state
        );

        // Convert NPU output to float
        std::vector<float> out_npu(out_sz);
        for (int i = 0; i < out_sz; i++) {
            out_npu[i] = to_float(buf_out[i]);
        }

        if (verbosity >= 2) {
            std::cout << "\nFirst 10 outputs:\n";
            std::cout << "CPU vs NPU:\n";
            for (int i = 0; i < std::min(10, seq_len); i++) {
                std::cout << i << ": " << out_cpu[i] << " vs " << out_npu[i] << "\n";
            }
        }

        pass = verify_results(out_cpu.data(), out_npu.data(), seq_len);
    }

    // Print result
    std::cout << "\n====================================\n";
    if (pass) {
        std::cout << "TEST PASSED!\n";
        std::cout << "====================================\n";
        return 0;
    } else {
        std::cout << "TEST FAILED!\n";
        std::cout << "====================================\n";
        return 1;
    }
}
