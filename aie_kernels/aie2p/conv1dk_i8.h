//===- conv1dk_i8.h ---------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef _CONV1DK_I8_H
#define _CONV1DK_I8_H

extern "C" {

/**
 * @brief 1D Convolution with int8 activations and weights
 * 
 * @param input      Input tensor [length, in_channels]
 * @param kernels    Convolution weights [kernel_size, in_channels, out_channels]
 * @param output     Output tensor [length, out_channels]
 * @param length     Sequence length 
 * @param in_channels  Number of input channels 
 * @param out_channels Number of output channels 
 * @param kernel_size  Size of convolution kernel
 * @param scale       Quantization scale factor for fixed-point conversion
 * 
 * @note Uses 'same' padding - output length equals input length
 */
//===- conv1dk_i8.h ---------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//


void conv1dk_i8(int8_t *input, int8_t *kernels, int8_t *output,
                const int32_t length, const int32_t in_channels,
                const int32_t out_channels, const int32_t kernel_size,
                const int32_t scale);

void conv1dk_i8_causal(int8_t *input, int8_t *kernels, int8_t *output,
                       const int32_t length, const int32_t in_channels,
                       const int32_t out_channels, const int32_t kernel_size,
                       const int32_t scale);

} // extern "C"

#endif // _CONV1DK_I8_H