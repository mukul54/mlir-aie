# Mamba Selective Scan on AIE

Implements the Mamba selective scan operation on AMD NPU.

## State Update Formula

```
state[n] = state[n] * exp(dt * A[n]) + B[t,n] * x[t] * dt
y[t] = sum(state[n] * C[t,n]) + D * x[t]
```

## Build & Run

```bash
# Default: seq_len=128, d_state=16, bf16
make clean && make
make run

# Custom parameters
make clean
make seq_len=256 d_state=16 dtype=bf16
make run seq_len=256
```

## Parameters

- `seq_len`: Sequence length (default: 128)
- `d_state`: State dimension (default: 16, only 16 supported currently)
- `dtype`: Data type (`bf16` or `f32`, default: bf16)

## Files

- `mamba_scan.py` - MLIR AIE design
- `mamba_scan.cc` - AIE kernel (in aie_kernels/aie2p/)
- `test.cpp` - Host test with CPU reference
- `Makefile` - Build configuration

## Performance

Expected throughput on NPU2: ~5-10 GOPS for seq_len=128, d_state=16
