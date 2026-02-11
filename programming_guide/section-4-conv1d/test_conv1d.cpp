// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h" 

constexpr int LENGTH = 128;
constexpr int IN_CHANNELS = 64;
constexpr int OUT_CHANNELS = 64;
constexpr int KERNEL_SIZE = 1;

constexpr int IN_SIZE = LENGTH * IN_CHANNELS;
constexpr int OUT_SIZE = LENGTH * OUT_CHANNELS;
constexpr int WEIGHTS_SIZE = KERNEL_SIZE * IN_CHANNELS * OUT_CHANNELS;

int main(int argc, char *argv[]) {
  // Load instructions using test_utils
  std::vector<uint32_t> instr_v = test_utils::load_instr_binary("build/insts.bin");
  int instr_size = instr_v.size();

  std::cout << "Sequence instr count: " << instr_size << std::endl;

  // Initialize XRT
  std::cout << "Opening device..." << std::endl;
  auto device = xrt::device(0);
  
  std::cout << "Loading xclbin..." << std::endl;
  auto xclbin = xrt::xclbin(std::string("build/final.xclbin"));
  
  std::cout << "Registering xclbin..." << std::endl;
  device.register_xclbin(xclbin);
  
  std::cout << "Getting kernel..." << std::endl;
  auto kernel = xrt::kernel(device, xclbin.get_uuid(), "MLIR_AIE");

  // Allocate buffers
    std::cout << "Allocating buffers..." << std::endl;
    auto bo_instr = xrt::bo(device, instr_size * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_input = xrt::bo(device, IN_SIZE * sizeof(int8_t), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(3));
    auto bo_weights = xrt::bo(device, WEIGHTS_SIZE * sizeof(int8_t), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(4));
    auto bo_output = xrt::bo(device, OUT_SIZE * sizeof(int8_t), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(5));

  // Map buffers
  std::cout << "Mapping buffers..." << std::endl;
  auto buf_input = bo_input.map<int8_t*>();
  auto buf_weights = bo_weights.map<int8_t*>();
  auto buf_output = bo_output.map<int8_t*>();
  auto buf_instr = bo_instr.map<uint32_t*>();

  // Fill input with test pattern
  for (int i = 0; i < IN_SIZE; i++) {
    buf_input[i] = (i % 10) - 5;
  }

  // Fill weights (identity for testing)
  for (int i = 0; i < WEIGHTS_SIZE; i++) {
    buf_weights[i] = 1;
  }

  // Clear output
  for (int i = 0; i < OUT_SIZE; i++) {
    buf_output[i] = 0;
  }

  // Copy instructions
  std::memcpy(buf_instr, instr_v.data(), instr_size * sizeof(int));

  // Sync buffers
  std::cout << "Syncing buffers..." << std::endl;
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Execute kernel
  std::cout << "Running kernel..." << std::endl;
  unsigned int opcode = 1; // Add the opcode
  auto run = kernel(opcode, bo_instr, instr_size, bo_input, bo_weights, bo_output);
  run.wait();

  // Get results
  std::cout << "Getting results..." << std::endl;
  bo_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Verify results
  std::cout << "Verifying results..." << std::endl;
  int errors = 0;
  
  // Constants for clamping
  const int32_t SMAX = 127;
  const int32_t SMIN = -128;
  
  for (int t = 0; t < LENGTH && errors < 10; t++) {
    for (int oc = 0; oc < OUT_CHANNELS && errors < 10; oc++) {
      
      int32_t expected_accum = 0;
      for (int ic = 0; ic < IN_CHANNELS; ic++) {
        expected_accum += buf_input[t * IN_CHANNELS + ic];
      }
      
      // --- CORRECTED LOGIC ---
      // Since scale=0, we only apply clamping, no shifting.
      int32_t expected = expected_accum; 
      expected = (expected > SMAX) ? SMAX : (expected < SMIN) ? SMIN : expected;
      // --- END OF LOGIC ---

      int actual = buf_output[t * OUT_CHANNELS + oc];
      
      if (actual != expected) {
        std::cout << "Mismatch at [" << t << ", " << oc << "]: "
                  << "expected=" << expected << ", actual=" << actual 
                  << " (accumulator was " << expected_accum << ")" << std::endl;
        errors++;
      }
    }
  }

  if (errors == 0) {
    std::cout << "PASS!" << std::endl;
    return 0;
  } else {
    std::cout << "FAIL!" << std::endl;
    return 1;
  }
}