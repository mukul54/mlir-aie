##===- Makefile: W4A8 GEMM (int4-packed B, int8 MMUL) — svdq_gemm v1 -----===##
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
subdir=whole_array_w4a8
targetname=whole_array_w4a8

M?=512
K?=512
N?=512
m?=32
k?=32
n?=32
n_aie_cols?=4
b_col_maj?=0
c_col_maj?=0

kernels=mm_w4a8_${m}x${k}x${n}
aieargs+=-m $m -k $k -n $n --n-aie-cols ${n_aie_cols} --b-col-maj ${b_col_maj}
runargs+=--b_col_maj ${b_col_maj}
target_suffix=${M}x${K}x${N}_${m}x${k}x${n}_${n_aie_cols}c
use_placed?=0
use_iron?=0
use_scalar?=0

use_linear_buf_alloc?=1
ifeq (${use_linear_buf_alloc}, 1)
buffer_aloc_flag=basic-sequential
else
buffer_aloc_flag=bank-aware
endif

KERNEL_DEFINES=-D${dtype_in}_${dtype_out}_ONLY -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -DVECTORIZED_ONLY

include ${srcdir}/../makefile-common

build/mm_w4a8_${m}x${k}x${n}.o: ${kernels_dir}/mm_w4a8.cc
	mkdir -p ${@D}
	cd ${@D} && ${KERNEL_CC} ${KERNEL_CFLAGS} ${KERNEL_DEFINES} -c $< -o ${@F}
