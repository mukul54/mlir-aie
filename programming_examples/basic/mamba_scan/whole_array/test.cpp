//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Test harness for Mamba Selective Scan - Whole Array
// Processes n_channels independent channels in parallel.
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
// CPU Reference Implementation (single channel)
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
    for (int n = 0; n < d_state; n++) {
        state[n] = 0.0f;
    }

    for (int t = 0; t < seq_len; t++) {
        float x_t = x[t];
        float dt_t = dt[t];
        float y_sum = 0.0f;

        for (int n = 0; n < d_state; n++) {
            float B_t = B[t * d_state + n];
            float C_t = C[t * d_state + n];

            float exp_dt_A = expf(dt_t * A[n]);
            state[n] = state[n] * exp_dt_A + B_t * x_t * dt_t;
            y_sum += state[n] * C_t;
        }

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

bool verify_channel(const float* cpu_out, const float* npu_out, int seq_len,
                    int channel_idx, float abs_tol = 0.5f, float rel_tol = 0.05f) {
    int errors = 0;
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;

    for (int i = 0; i < seq_len; i++) {
        float abs_err = std::abs(cpu_out[i] - npu_out[i]);
        float rel_err = abs_err / (std::abs(cpu_out[i]) + 1e-8f);

        max_abs_error = std::max(max_abs_error, abs_err);
        max_rel_error = std::max(max_rel_error, rel_err);

        if (abs_err > abs_tol && rel_err > rel_tol) {
            if (errors < 5) {
                std::cerr << "  Channel " << channel_idx << " error at t=" << i
                         << ": CPU=" << cpu_out[i] << " NPU=" << npu_out[i]
                         << " abs=" << abs_err << " rel=" << rel_err << "\n";
            }
            errors++;
        }
    }

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
    int n_channels = 32;  // default: 8 cols * 4 rows (full NPU2)
    bool do_verify = true;
    int n_iterations = 1;
    int verbosity = 1;

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
        } else if (arg == "--n_channels" && i + 1 < argc) {
            n_channels = std::atoi(argv[++i]);
        } else if (arg == "--verify") {
            do_verify = std::atoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            n_iterations = std::atoi(argv[++i]);
        } else if (arg == "-v" && i + 1 < argc) {
            verbosity = std::atoi(argv[++i]);
        }
    }

    if (verbosity >= 1) {
        std::cout << "============================================\n";
        std::cout << "Mamba Selective Scan - Whole Array Test\n";
        std::cout << "============================================\n";
        std::cout << "Sequence length: " << seq_len << "\n";
        std::cout << "State dimension: " << d_state << "\n";
        std::cout << "Channels:        " << n_channels << "\n";
        std::cout << "============================================\n";
    }

    srand(42);

    // Per-channel buffer sizes (matching mamba_scan_whole_array.py layout)
    int x_sz = seq_len;
    int dt_sz = seq_len;
    int A_sz = d_state;
    int D_sz = 2;  // D + padding
    int params_per_channel = x_sz + dt_sz + A_sz + D_sz;

    int B_sz = seq_len * d_state;
    int C_sz = seq_len * d_state;
    int B_C_sz = B_sz + C_sz;

    int out_per_channel = seq_len;

    // Total buffer sizes
    int total_params_sz = n_channels * params_per_channel;
    int total_out_sz = n_channels * out_per_channel;

    // Load instructions
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

    // Allocate buffers matching runtime_sequence(all_params, B_C, all_out)
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                           XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_params = xrt::bo(device, total_params_sz * sizeof(DATATYPE),
                            XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_B_C = xrt::bo(device, B_C_sz * sizeof(DATATYPE),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_out = xrt::bo(device, total_out_sz * sizeof(DATATYPE),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    if (verbosity >= 1) {
        std::cout << "Generating test data for " << n_channels << " channels...\n";
    }

    DATATYPE *buf_params = bo_params.map<DATATYPE*>();
    DATATYPE *buf_B_C = bo_B_C.map<DATATYPE*>();
    DATATYPE *buf_out = bo_out.map<DATATYPE*>();

    // CPU-side storage per channel (for verification)
    std::vector<std::vector<float>> x_cpu(n_channels, std::vector<float>(x_sz));
    std::vector<std::vector<float>> dt_cpu(n_channels, std::vector<float>(dt_sz));
    std::vector<std::vector<float>> A_cpu(n_channels, std::vector<float>(A_sz));
    std::vector<float> D_cpu(n_channels);

    // Shared B, C (same for all channels, as in the Mamba architecture)
    std::vector<float> B_cpu(B_sz);
    std::vector<float> C_cpu(C_sz);

    // Pack per-channel params into all_params buffer
    // Layout: [ch0_params, ch1_params, ..., ch(n-1)_params]
    // Channel ordering: column-major (col0_row0, col0_row1, ..., col0_row3, col1_row0, ...)
    for (int ch = 0; ch < n_channels; ch++) {
        int base = ch * params_per_channel;
        int offset = 0;

        // x values
        for (int i = 0; i < x_sz; i++) {
            x_cpu[ch][i] = get_random<float>() * 0.5f;
            buf_params[base + offset + i] = DATATYPE(x_cpu[ch][i]);
        }
        offset += x_sz;

        // dt values (small positive)
        for (int i = 0; i < dt_sz; i++) {
            dt_cpu[ch][i] = std::abs(get_random<float>()) * 0.1f + 0.01f;
            buf_params[base + offset + i] = DATATYPE(dt_cpu[ch][i]);
        }
        offset += dt_sz;

        // A values (negative for stability)
        for (int i = 0; i < A_sz; i++) {
            A_cpu[ch][i] = -std::abs(get_random<float>()) * 0.5f;
            buf_params[base + offset + i] = DATATYPE(A_cpu[ch][i]);
        }
        offset += A_sz;

        // D value + padding
        D_cpu[ch] = get_random<float>() * 0.5f;
        buf_params[base + offset] = DATATYPE(D_cpu[ch]);
        buf_params[base + offset + 1] = DATATYPE(0.0f);
    }

    // Pack shared B_C buffer: [B_flat, C_flat]
    int offset = 0;
    for (int i = 0; i < B_sz; i++) {
        B_cpu[i] = get_random<float>() * 0.3f;
        buf_B_C[offset + i] = DATATYPE(B_cpu[i]);
    }
    offset += B_sz;
    for (int i = 0; i < C_sz; i++) {
        C_cpu[i] = get_random<float>() * 0.3f;
        buf_B_C[offset + i] = DATATYPE(C_cpu[i]);
    }

    memset(buf_out, 0, total_out_sz * sizeof(DATATYPE));

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
        std::cout << "Running kernel on " << n_channels << " channels...\n";
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < n_iterations; iter++) {
        unsigned int opcode = 3;
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
        double gops = (2.0 * n_channels * seq_len * d_state) / (kernel_time_ms * 1e6);
        std::cout << "Throughput: " << gops << " GOPS (" << n_channels << " channels)\n";
    }

    // Verification: run CPU reference for each channel
    bool pass = true;
    if (do_verify) {
        if (verbosity >= 1) {
            std::cout << "Verifying " << n_channels << " channels...\n";
        }

        int channels_passed = 0;

        for (int ch = 0; ch < n_channels; ch++) {
            std::vector<float> state_cpu(d_state);
            std::vector<float> out_cpu(out_per_channel);

            mamba_scan_cpu_reference(
                x_cpu[ch].data(), dt_cpu[ch].data(), A_cpu[ch].data(),
                B_cpu.data(), C_cpu.data(), D_cpu[ch],
                state_cpu.data(), out_cpu.data(),
                seq_len, d_state
            );

            // Extract NPU output for this channel
            int out_base = ch * out_per_channel;
            std::vector<float> out_npu(out_per_channel);
            for (int i = 0; i < out_per_channel; i++) {
                out_npu[i] = to_float(buf_out[out_base + i]);
            }

            if (verbosity >= 2 && ch < 2) {
                std::cout << "\nChannel " << ch << " first 5 outputs (CPU vs NPU):\n";
                for (int i = 0; i < std::min(5, seq_len); i++) {
                    std::cout << "  t=" << i << ": " << out_cpu[i]
                             << " vs " << out_npu[i] << "\n";
                }
            }

            bool ch_pass = verify_channel(
                out_cpu.data(), out_npu.data(), seq_len, ch
            );

            if (ch_pass) {
                channels_passed++;
            } else {
                pass = false;
            }
        }

        if (verbosity >= 1) {
            std::cout << "Channels passed: " << channels_passed
                     << " / " << n_channels << "\n";
        }
    }

    // Print result
    std::cout << "\n============================================\n";
    if (pass) {
        std::cout << "TEST PASSED! (" << n_channels << " channels)\n";
        std::cout << "============================================\n";
        return 0;
    } else {
        std::cout << "TEST FAILED!\n";
        std::cout << "============================================\n";
        return 1;
    }
}
