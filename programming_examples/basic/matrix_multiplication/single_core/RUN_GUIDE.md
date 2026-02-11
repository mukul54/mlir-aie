# Matrix Multiplication Single Core - Complete Run Guide

This guide shows you how to build, run, and trace the matrix multiplication example on AMD NPU.

---

## **Prerequisites**

1. **AMD Ryzen AI system** with NPU (Phoenix, Hawk, or Strix)
2. **MLIR-AIE tools installed** (`aiecc`, `xclbinutil`, etc.)
3. **XRT runtime** installed
4. **Peano compiler** (llvm-aie) or Chess compiler

Check your setup:
```bash
which aiecc.py
xbutil examine  # Should show your NPU device
```

---

## **Quick Start - Default Build**

### **1. Build Everything (MLIR + Kernel + Host)**

```bash
cd /home/mr28/projects/my_mlir/mlir-aie/programming_examples/basic/matrix_multiplication/single_core

# Build with default parameters (512x512x512, tile=32x32x32, dtype=i16)
make clean
make
```

**This will:**
- Generate MLIR from Python: `single_core.py` → `build/aie_512x512x512_32x32x32.mlir`
- Compile AIE kernel: `mm.cc` → `build/mm_32x32x32.o`
- Build xclbin: → `build/final_512x512x512_32x32x32.xclbin`
- Compile host test: `test.cpp` → `single_core.exe`

### **2. Run the Test**

```bash
make run
```

**Expected output:**
```
Matrix size 512x512x512
Sequence instr count: ...
PASS!
Kernel time: X.XXX ms
Throughput: XXX GOPS
```

---

## **Custom Matrix Sizes**

### **Small Test (Fast)**

```bash
make clean
make M=128 K=128 N=128 m=32 k=32 n=32
make run M=128 K=128 N=128
```

**Parameters:**
- `M, K, N` = Full matrix dimensions (M×K @ K×N = M×N)
- `m, k, n` = Tile dimensions processed by single core

**Constraints:**
- `M % m == 0`, `K % k == 0`, `N % n == 0`
- `m % r == 0`, `k % s == 0`, `n % t == 0` where (r,s,t) = hardware intrinsic size

For i16 on NPU1: (r,s,t) = (4,4,4)

### **Large Matrix**

```bash
make clean
make M=1024 K=1024 N=1024 m=64 k=64 n=64
make run M=1024 K=1024 N=1024
```

---

## **Different Data Types**

### **BFloat16 (Faster, Less Precise)**

```bash
make clean
make dtype_in=bf16 dtype_out=bf16 M=512 K=512 N=512 m=64 k=64 n=64
make run dtype_in=bf16 dtype_out=bf16 M=512 K=512 N=512
```

### **INT8 (Fastest)**

```bash
make clean
make dtype_in=i8 dtype_out=i32 M=512 K=512 N=512 m=64 k=64 n=64
make run dtype_in=i8 dtype_out=i32 M=512 K=512 N=512
```

**Available types:**
- `dtype_in`: `i8`, `i16`, `bf16`
- `dtype_out`: `i8`, `i16`, `i32`, `bf16`, `f32`

---

## **Enable Tracing (Performance Analysis)**

### **1. Build with Tracing Enabled**

```bash
make clean
# Set trace_size > 0 to enable tracing
make trace_size=8192 M=256 K=256 N=256 m=32 k=32 n=32
```

This creates `build/trace_256x256x256_32x32x32.xclbin` with tracing instrumentation.

### **2. Run with Trace Output**

```bash
./single_core.exe \
  --xclbin build/final_256x256x256_32x32x32.xclbin \
  --instr build/insts_256x256x256_32x32x32.txt \
  --kernel MLIR_AIE \
  --trace_sz 8192 \
  --trace trace.txt \
  --verify 1 \
  --iters 1 \
  --M 256 --K 256 --N 256
```

**Parameters:**
- `--trace trace.txt` - Output file for trace data
- `--trace_sz 8192` - Size of trace buffer (must match build)
- `--verify 1` - Verify results against CPU golden reference
- `--iters 1` - Number of iterations to run

### **3. Parse Trace Output**

```bash
# Parse raw trace to human-readable format
python ${MLIR_AIE}/utils/parse_eventIR.py \
  --filename trace.txt \
  --mlir build/aie_256x256x256_32x32x32.mlir \
  --colshift 1
```

**Trace shows:**
- DMA transfer events
- Compute core activity
- Memory stalls
- Lock contention
- Instruction events

### **4. Visualize with Perfetto**

```bash
# Generate Chrome trace format
python ${MLIR_AIE}/utils/trace_events_to_chrome.py \
  --filename trace.txt \
  --output trace.json

# Open trace.json in Chrome at chrome://tracing
```

---

## **Alternative Design Variants**

### **1. Placed Design (Explicit Tile Placement)**

```bash
make clean
make use_placed=1
make run
```

Uses `single_core_placed.py` which has explicit `dma_configure_task_for` operations instead of `npu_dma_memcpy_nd`.

### **2. IRON Design (High-Level API)**

```bash
make clean
make use_iron=1
make run
```

Uses `single_core_iron.py` - newer high-level API with automatic placement.

---

## **Column-Major Matrix B**

By default, B is row-major. To use column-major:

```bash
make clean
make b_col_maj=1 M=256 K=256 N=256
make run b_col_maj=1 M=256 K=256 N=256
```

This changes the data layout transformation in ObjectFIFOs.

---

## **Performance Tuning**

### **1. Use Chess Compiler (Potentially Faster)**

```bash
make clean
make use_chess=1
```

Chess is AMD's optimizing AIE compiler (alternative to Peano/llvm-aie).

### **2. Enable Peano Optimizations**

```bash
make clean
make opt_perf=1
```

Enables aggressive loop optimizations in kernel (increases compile time).

### **3. Bank-Aware Buffer Allocation**

```bash
make clean
make use_linear_buf_alloc=0
```

Optimizes memory bank usage to reduce conflicts (default is linear).

---

## **Debugging**

### **1. View Generated MLIR**

```bash
make build/aie_512x512x512_32x32x32.mlir
cat build/aie_512x512x512_32x32x32.mlir
```

### **2. Verbose Build**

```bash
make V=1
```

Shows all compilation commands.

### **3. Test with Verification**

```bash
./single_core.exe \
  --xclbin build/final_512x512x512_32x32x32.xclbin \
  --instr build/insts_512x512x512_32x32x32.txt \
  --kernel MLIR_AIE \
  --verify 1 \
  --verbosity 2 \
  --M 512 --K 512 --N 512
```

`--verbosity 2` prints input matrices and element-wise comparison.

### **4. Run Only Host Code (CPU Reference)**

Modify `test.cpp` to run CPU matmul without NPU:

```cpp
// Comment out the kernel run, just do verification
matmul_common::do_matmul_cpu<A_DATATYPE, C_DATATYPE, ACC_DATATYPE>(
    AVec.data(), BVec.data(), CVec.data(), M, K, N, false, b_col_maj, c_col_maj);
```

---

## **Complete Example with Tracing**

```bash
#!/bin/bash
# Complete build and trace workflow

cd /home/mr28/projects/my_mlir/mlir-aie/programming_examples/basic/matrix_multiplication/single_core

# Clean previous build
make clean

# Build with tracing
echo "Building with tracing enabled..."
make trace_size=16384 M=256 K=256 N=256 m=64 k=64 n=32 dtype_in=i16 dtype_out=i32

# Run test with trace capture
echo "Running test..."
./single_core.exe \
  --xclbin build/final_256x256x256_64x64x32.xclbin \
  --instr build/insts_256x256x256_64x64x32.txt \
  --kernel MLIR_AIE \
  --trace_sz 16384 \
  --trace trace_output.txt \
  --verify 1 \
  --iters 5 \
  --warmup 2 \
  --verbosity 1 \
  --M 256 --K 256 --N 256

# Parse trace
echo "Parsing trace data..."
python ${MLIR_AIE}/utils/parse_eventIR.py \
  --filename trace_output.txt \
  --mlir build/aie_256x256x256_64x64x32.mlir \
  --colshift 1

echo "Done! Check trace_output.txt for results"
```

---

## **Understanding the Output**

### **Performance Metrics**

```
Kernel time: 2.345 ms
Throughput: 45.6 GOPS
```

**Calculations:**
- Total operations = 2 × M × K × N (for matrix multiply)
- GOPS = (2 × M × K × N) / (kernel_time_ms × 1e6)
- For 256³: 2 × 256³ = 33.5 million ops → ~14.3 GOPS @ 2.345ms

### **Trace Events**

Example trace output:
```
[Tile(0,2)] PORT_RUNNING_0 at cycle 1234    # Input A DMA started
[Tile(0,2)] INSTR_VECTOR at cycle 1240      # Vector instruction executed
[Tile(0,2)] MEMORY_STALL at cycle 1250      # Stalled waiting for memory
[Tile(0,2)] PORT_RUNNING_2 at cycle 2000    # Output C DMA started
```

**Common events:**
- `PORT_RUNNING_X` - DMA channel X is active
- `INSTR_VECTOR` - SIMD instruction executed
- `INSTR_EVENT_X` - Custom instrumentation point
- `MEMORY_STALL` - Core waiting for memory
- `LOCK_STALL` - Core waiting for lock

---

## **Common Issues**

### **1. Build Fails: "Module Not Found"**

```bash
# Set Python path
export PYTHONPATH=${MLIR_AIE}/python:$PYTHONPATH
```

### **2. Runtime Fails: "No Devices Found"**

```bash
# Check XRT installation
xbutil examine

# Reload drivers
sudo rmmod amdxdna
sudo modprobe amdxdna
```

### **3. Wrong Results / Verification Failed**

- Check matrix dimensions satisfy constraints (M%m==0, etc.)
- Verify data types match between Python, C++ kernel, and host
- Try smaller matrix first (128×128×128)

### **4. Trace Buffer Too Small**

```
Error: Trace buffer overflow
```

Increase `trace_size`:
```bash
make clean
make trace_size=32768  # Double the size
```

---

## **Next Steps**

1. **Experiment with sizes** to find performance sweet spot
2. **Enable tracing** to identify bottlenecks
3. **Try different data types** (bf16 is often faster than i16)
4. **Read the whole_array example** to see multi-core scaling
5. **Check the Mamba example** (in this same folder!) for sequence models

---

## **File Structure After Build**

```
single_core/
├── single_core.py              # Python MLIR generator
├── test.cpp                    # Host test code
├── Makefile                    # Build configuration
├── build/
│   ├── aie_512x512x512_32x32x32.mlir      # Generated MLIR
│   ├── mm_32x32x32.o                       # Compiled kernel
│   ├── final_512x512x512_32x32x32.xclbin  # NPU binary
│   ├── insts_512x512x512_32x32x32.txt     # DMA instructions
│   └── trace_*.xclbin                      # Traced version (if enabled)
└── single_core.exe             # Host executable
```

---

## **Resources**

- [MLIR-AIE Documentation](https://xilinx.github.io/mlir-aie/)
- [AIE API Reference](https://xilinx.github.io/aie_api/)
- [XRT Documentation](https://xilinx.github.io/XRT/)
- [Whole Array Example](../whole_array/README.md) - Multi-core version

For questions or issues, see: https://github.com/Xilinx/mlir-aie/issues
