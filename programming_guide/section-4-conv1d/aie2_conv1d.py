# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.

import sys
import numpy as np

from aie.iron import (
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.controlflow import range_


def conv1d_design(dev, length, in_channels, out_channels, kernel_size):
    """
    Simple Conv1D test design
    
    Args:
        dev: Device (NPU1Col1 or NPU2Col1)
        length: Sequence length (must be multiple of 4)
        in_channels: Input channels (must be multiple of 8)
        out_channels: Output channels (must be multiple of 8)
        kernel_size: Convolution kernel size (1, 3, 4, etc.)
    """
    
    # Calculate sizes
    tile_length = 32  # Process 32 positions at a time
    
    actIn = tile_length * in_channels
    bufIn = actIn * 2
    
    weights_size = kernel_size * in_channels * out_channels
    
    actOut = tile_length * out_channels
    bufOut = actOut * 2
    
    tensorInSize = length * in_channels
    tensorOutSize = length * out_channels
    
    num_tiles = length // tile_length
    
    # Type definitions
    actIn_ty = np.ndarray[(actIn,), np.dtype[np.int8]]
    bufIn_ty = np.ndarray[(bufIn,), np.dtype[np.int8]]
    weights_ty = np.ndarray[(weights_size,), np.dtype[np.int8]]
    out_ty = np.ndarray[(actOut,), np.dtype[np.int8]]
    bufOut_ty = np.ndarray[(bufOut,), np.dtype[np.int8]]
    tensorIn_ty = np.ndarray[(tensorInSize,), np.dtype[np.int8]]
    tensorOut_ty = np.ndarray[(tensorOutSize,), np.dtype[np.int8]]
    
    # Kernel declaration
    conv1dk_i8_kernel = Kernel(
        "conv1dk_i8",
        "conv1dk_i8.o",
        [
            actIn_ty,
            weights_ty,
            out_ty,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )
    
    # ObjectFifos
    of_inOF_act_L3L2 = ObjectFifo(bufIn_ty, name="inOF_act_L3L2")
    of_act_L2_02 = of_inOF_act_L3L2.cons().forward(obj_type=actIn_ty, name="act_L2_02")
    
    of_inOF_wts_0_L3L2 = ObjectFifo(weights_ty, depth=1, name="inOF_wts_0_L3L2")
    
    of_out_02_L2 = ObjectFifo(out_ty, name="out_02_L2")
    of_outOFL2L3 = of_out_02_L2.cons().forward(obj_type=bufOut_ty, name="outOFL2L3")
    
    # Core function
    def core_fn(of_wts, of_act, of_out, conv1dk_i8):
        tile_len = tile_length
        ci = in_channels
        co = out_channels
        ks = kernel_size
        scale = 0  # No scaling for simple test
        
        elemWts = of_wts.acquire(1)
        
        for _ in range_(num_tiles):
            elemIn = of_act.acquire(1)
            elemOut = of_out.acquire(1)
            
            conv1dk_i8(elemIn, elemWts, elemOut, tile_len, ci, co, ks, scale)
            
            of_act.release(1)
            of_out.release(1)
        
        of_wts.release(1)
    
    # Worker
    worker = Worker(
        core_fn,
        [
            of_inOF_wts_0_L3L2.cons(),
            of_act_L2_02.cons(),
            of_out_02_L2.prod(),
            conv1dk_i8_kernel,
        ],
        stack_size=0x600,
    )
    
    # Runtime
    rt = Runtime()
    with rt.sequence(tensorIn_ty, weights_ty, tensorOut_ty) as (I, W, O):
        rt.start(worker)
        rt.fill(of_inOF_act_L3L2.prod(), I)
        rt.fill(of_inOF_wts_0_L3L2.prod(), W)
        rt.drain(of_outOFL2L3.cons(), O, wait=True)
    
    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    device_name = str(sys.argv[1])
    if device_name == "npu":
        dev = NPU1Col1()
    elif device_name == "npu2":
        dev = NPU2Col1()
    else:
        raise ValueError(f"[ERROR] Device name {device_name} is unknown")
    
    # Test configuration
    length = 128  # Must be multiple of 32
    in_channels = 64
    out_channels = 64
    kernel_size = 1  # Start with 1x1 conv for testing
    
    module = conv1d_design(dev, length, in_channels, out_channels, kernel_size)
    print(module)