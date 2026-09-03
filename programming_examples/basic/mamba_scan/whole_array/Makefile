##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
targetname=mamba_scan_whole_array

seq_len?=128
d_state?=16
dtype?=bf16
dev?=npu2
n_aie_cols?=8

n_aie_rows=4
n_channels=$(shell echo $$(($(n_aie_cols) * $(n_aie_rows))))

cpt?=1
aieargs+=--dev $(dev) --seq_len $(seq_len) --d_state $(d_state) --dtype $(dtype) --n-aie-cols $(n_aie_cols) --channels-per-tile $(cpt)

KERNEL_DEFINES=-DDTYPE=$(dtype)

include ${srcdir}/../../../makefile-common

# Auto-detect PEANO_INSTALL_DIR if not set (from mlir-aie Python package)
PEANO_INSTALL_DIR ?= $(shell python3 -c "from aie.utils.config import root_path; import os; print(os.path.normpath(os.path.join(root_path(), '..', 'llvm-aie')))")

# Runtime library directories
aie2_runtime_dir = ${MLIR_AIE_DIR}/aie_runtime_lib/AIE2
aie2p_runtime_dir = ${MLIR_AIE_DIR}/aie_runtime_lib/AIE2P

# Set device-specific variables based on dev parameter
ifeq ($(dev),npu2)
    devicename?=npu2
    kernels_dir=${srcdir}/../../../../aie_kernels/aie2p
    KERNEL_CFLAGS=${PEANOWRAP2P_FLAGS}
    runtime_dir=${aie2p_runtime_dir}
else
    devicename?=npu
    kernels_dir=${srcdir}/../../../../aie_kernels/aie2
    KERNEL_CFLAGS=${PEANOWRAP2_FLAGS}
    runtime_dir=${aie2_runtime_dir}
endif

target_suffix?=${seq_len}_${d_state}_${dtype}_${n_aie_cols}c_${cpt}cpt
mlir_target?=build/aie_${target_suffix}.mlir
xclbin_target?=build/final_${target_suffix}.xclbin
insts_target?=build/insts_${target_suffix}.txt

# KERNEL_CC defines the C compiler used to compile the kernel.
KERNEL_CC=${PEANO_INSTALL_DIR}/bin/clang++

.PHONY: all
all: ${xclbin_target} ${targetname}.exe

# Compile the mamba_scan kernel with include path to runtime lib for lut_based_ops.h
build/mamba_scan.o: ${kernels_dir}/mamba_scan.cc
	mkdir -p ${@D}
	cd ${@D} && ${KERNEL_CC} ${KERNEL_CFLAGS} ${KERNEL_DEFINES} -I${runtime_dir} -c $< -o ${@F}

# Compile the LUT lookup tables (required for exp function)
build/lut_based_ops.o: ${runtime_dir}/lut_based_ops.cpp
	mkdir -p ${@D}
	cd ${@D} && ${KERNEL_CC} ${KERNEL_CFLAGS} -I. -c $< -o ${@F}

# Bundle kernel and LUT tables into a static library
build/kernels.a: build/mamba_scan.o build/lut_based_ops.o
	ar rvs $@ $+

${mlir_target}: ${srcdir}/mamba_scan_whole_array.py
	mkdir -p ${@D}
	python3 $< ${aieargs} > $@

${xclbin_target}: ${mlir_target} build/kernels.a
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
		--aie-generate-npu-insts --npu-insts-name=${insts_target:build/%=%} $(<:%=../%)

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@
endif

.PHONY: run
run: ${targetname}.exe ${xclbin_target}
	${powershell} ./${targetname}.exe --xclbin ${xclbin_target} --instr ${insts_target} \
		--kernel MLIR_AIE --seq_len $(seq_len) --d_state $(d_state) \
		--n_channels $(n_channels) --verify 1 -v 2

.PHONY: clean
clean:
	rm -rf build _build ${targetname}.exe
