# Whole Array vs Single Core: Complete Breakdown

This document explains the architectural and implementation differences between the **single_core** and **whole_array** matrix multiplication designs.

---

## **TL;DR - Key Differences**

| Aspect | Single Core | Whole Array |
|--------|-------------|-------------|
| **Compute cores** | 1 core (tile 0,2) | 4-32 cores (4 rows × 1-8 cols) |
| **Speedup** | 1× (baseline) | 4-32× (scales with cores) |
| **Complexity** | Simple (371 lines) | Complex (627 lines) |
| **ObjectFIFOs** | 6 FIFOs | 40+ FIFOs (for 4×4 array) |
| **Data distribution** | Sequential | **Broadcast + Distribute** |
| **Use case** | Debugging, learning | Production, performance |

---

## **Part 1: Architecture Comparison**

### **Single Core Architecture**

```
┌─────────────────────────────────────────────────┐
│  Shim Tile (0,0)                                │
│  ┌────────┐  ┌────────┐  ┌────────┐           │
│  │  inA   │  │  inB   │  │  outC  │           │
│  └────┬───┘  └────┬───┘  └────▲───┘           │
└───────┼──────────┼───────────┼─────────────────┘
        │          │           │
┌───────┼──────────┼───────────┼─────────────────┐
│  Mem Tile (0,1)  │           │                 │
│  ┌────▼───┐  ┌──▼────┐  ┌───┴────┐           │
│  │  memA  │  │ memB  │  │  memC  │           │
│  └────┬───┘  └───┬───┘  └───▲────┘           │
└───────┼──────────┼───────────┼─────────────────┘
        │          │           │
┌───────┼──────────┼───────────┼─────────────────┐
│  Compute Tile (0,2)          │                 │
│  ┌────▼───────────▼──────────┴────┐            │
│  │  MATMUL KERNEL (mm.cc)         │            │
│  │  Processes ALL tiles           │            │
│  │  sequentially                  │            │
│  └────────────────────────────────┘            │
└─────────────────────────────────────────────────┘
```

**Data flow:** DRAM → Shim → Mem → Single Compute Core → Mem → Shim → DRAM

---

### **Whole Array Architecture (4×4 example)**

```
                        COLUMNS
                  0       1       2       3
              ┌───────┌───────┌───────┌───────┐
    Shim (0)  │ inA0  │ inA1  │ inA2  │ inA3  │  ← Each shim handles
              │ inB0  │ inB1  │ inB2  │ inB3  │    different columns
              │ outC0 │ outC1 │ outC2 │ outC3 │    of B and C
              └───┬───┴───┬───┴───┬───┴───┬───┘
                  │       │       │       │
              ┌───▼───┌───▼───┌───▼───┌───▼───┐
    Mem (1)   │ memA0 │ memA1 │ memA2 │ memA3 │  ← Mem tiles
              │ memB0 │ memB1 │ memB2 │ memB3 │    distribute data
              │ memC0 │ memC1 │ memC2 │ memC3 │    to compute cores
              └───┬───┴───┬───┴───┬───┴───┬───┘
                  │ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ │
                  │  X   X   X   X  │           ← BROADCAST!
                  │ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ │
              ┌───▼───┬───▼───┬───▼───┬───▼───┐
       R   2  │Core00 │Core01 │Core02 │Core03 │  ← Row 0
       O   3  │Core10 │Core11 │Core12 │Core13 │  ← Row 1
       W   4  │Core20 │Core21 │Core22 │Core23 │  ← Row 2
       S   5  │Core30 │Core31 │Core32 │Core33 │  ← Row 3
              └───────┴───────┴───────┴───────┘

Each core computes: C[row_block, col_block] = A[row_block, :] @ B[:, col_block]
```

**Data flow:**
- **A matrix:** Shim → Mem → **BROADCAST** across columns → All cores in same row
- **B matrix:** Shim → Mem → **BROADCAST** across rows → All cores in same column
- **C matrix:** Each core → **JOIN** in Mem tile → Shim → DRAM

---

## **Part 2: Critical New Concepts in Whole Array**

### **1. Multi-Dimensional Tile Arrays (Lines 231-237)**

**Single Core:**
```python
shim_tile = tile(0, 0)
mem_tile = tile(0, 1)
compute_tile = tile(0, 2)
```

**Whole Array:**
```python
# 2D array of tiles: tiles[row][col]
tiles = [
    [tile(col, row) for col in range(0, n_aie_cols)]
    for row in range(0, 6)  # Rows 0-5
]
shim_tiles = tiles[0]      # Row 0: [tile(0,0), tile(1,0), tile(2,0), tile(3,0)]
mem_tiles = tiles[1]       # Row 1: [tile(0,1), tile(1,1), tile(2,1), tile(3,1)]
core_tiles = tiles[2:]     # Rows 2-5 (4 rows of compute cores)

# Access specific core:
core_tiles[2][3]  # Core at row 2 (physical row 4), col 3
```

**Why?** Managing 16+ tiles individually is impractical. 2D array makes it scalable.

---

### **2. ObjectFIFO Broadcasting (Lines 263-284)**

**Single Core:**
```python
# Point-to-point: one producer, one consumer
memA = object_fifo("memA", mem_tile, compute_tile, 2, A_l1_ty)
```

**Whole Array:**
```python
# One-to-Many BROADCAST: one producer, MULTIPLE consumers
A_l2l1_fifos[row] = object_fifo(
    f"A_L2L1_{row}",
    mem_tiles[row // n_A_tiles_per_shim],  # One mem tile (producer)
    core_tiles[row][0:n_aie_cols],         # ALL cores in this row (consumers)
    #                ^^^^^^^^^^^^^^^^ BROADCAST to 4 cores!
    fifo_depth,
    A_l1_ty,
    [transformation]
)
```

**What happens:**
```
memA[row=0] broadcasts to:
  - core_tiles[0][0]  ← Column 0, gets same A tile
  - core_tiles[0][1]  ← Column 1, gets same A tile
  - core_tiles[0][2]  ← Column 2, gets same A tile
  - core_tiles[0][3]  ← Column 3, gets same A tile

All 4 cores in row 0 receive IDENTICAL A data!
```

**Why broadcast A?** Because in C = A @ B, **each row of A** is used with **all columns of B**:

```
C[0:64, 0:32]   = A[0:64, :] @ B[:, 0:32]    ← Row 0 of A
C[0:64, 32:64]  = A[0:64, :] @ B[:, 32:64]   ← SAME row 0 of A!
C[0:64, 64:96]  = A[0:64, :] @ B[:, 64:96]   ← SAME row 0 of A!
C[0:64, 96:128] = A[0:64, :] @ B[:, 96:128]  ← SAME row 0 of A!
```

---

### **3. ObjectFIFO Distribute (Lines 287-303)**

**What:** Split one large buffer into multiple smaller buffers sent to different consumers.

```python
# A_l3l2_fifos contains (m*k*n_A_tiles_per_shim) elements
# Need to split into n_A_tiles_per_shim separate (m*k) tiles

object_fifo_link(
    A_l3l2_fifos[i],                               # Producer: one large buffer
    [A_l2l1_fifos[j] for j in range(start, stop)], # Consumers: multiple FIFOs
    [],                                            # No offsets for producer
    of_offsets,  # Offsets for consumers: [0, m*k, 2*m*k, 3*m*k]
)
```

**Example with n_A_tiles_per_shim = 2:**

```
A_l3l2_fifos[0] contains one buffer of size (2 * m * k):
┌─────────────────────────────────┐
│  Tile 0 (m×k)  │  Tile 1 (m×k)  │  Total: 2*m*k elements
└─────────────────────────────────┘

Distribute to:
  A_l2l1_fifos[0] ← Gets Tile 0 (offset 0)
  A_l2l1_fifos[1] ← Gets Tile 1 (offset m*k)
```

**Visually:**

```
Before (in A_l3l2_fifos):
[a00, a01, ..., a_m-1,k-1,  ← Tile 0
 am0, am1, ..., a_2m-1,k-1] ← Tile 1

After distribute:
  A_l2l1_fifos[0]: [a00, a01, ..., a_m-1,k-1]      ← Row 0 cores
  A_l2l1_fifos[1]: [am0, am1, ..., a_2m-1,k-1]     ← Row 1 cores
```

---

### **4. ObjectFIFO Join (Lines 378-387)**

**What:** Opposite of distribute - **combine** multiple producer buffers into one consumer buffer.

```python
# Join output from 4 cores in a column into one buffer for shim
object_fifo_link(
    [C_l1l2_fifos[j][col] for j in range(n_aie_rows)],  # Multiple producers
    C_l2l3_fifos[col],                                   # One consumer
    of_offsets,  # [0, m*n, 2*m*n, 3*m*n]
    [],
)
```

**Example with n_aie_rows = 4:**

```
Input from 4 cores in column 0:
  C_l1l2_fifos[0][0]: [c00, c01, ...]  ← Core[0,0] output (m×n)
  C_l1l2_fifos[1][0]: [cm0, cm1, ...]  ← Core[1,0] output (m×n)
  C_l1l2_fifos[2][0]: [c2m0, c2m1, ...] ← Core[2,0] output (m×n)
  C_l1l2_fifos[3][0]: [c3m0, c3m1, ...] ← Core[3,0] output (m×n)

Join into C_l2l3_fifos[0] (size 4*m*n):
┌──────────────┬──────────────┬──────────────┬──────────────┐
│  Core[0,0]   │  Core[1,0]   │  Core[2,0]   │  Core[3,0]   │
│  offset 0    │  offset m*n  │  offset 2m*n │  offset 3m*n │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

**Why join?** The shim tile expects one contiguous buffer to write to DRAM, but we have 4 separate outputs from 4 cores.

---

### **5. Per-Core Workload Distribution (Lines 181, 404-408)**

**Single Core:**
```python
# One core processes ALL tiles sequentially
for _ in range_(tiles):  # tiles = M_div_m * N_div_n = 32 tiles
    # Process one output tile at a time
```

**Whole Array (4×4 = 16 cores):**
```python
# Each core processes FEWER tiles in parallel
n_tiles_per_core = (M // m) * (N // n) // n_aie_cores
# = 32 tiles / 16 cores = 2 tiles per core

for _ in range_(n_tiles_per_core):  # Each core: only 2 iterations!
    # All 16 cores run this loop in PARALLEL
```

**Workload distribution:**

```
Single Core (sequential):
Time 0: Core[0,0] processes tile 0
Time 1: Core[0,0] processes tile 1
...
Time 31: Core[0,0] processes tile 31
Total time: 32 units

Whole Array (parallel, 16 cores):
Time 0: Core[0,0] → tile 0,  Core[0,1] → tile 1,  ... Core[3,3] → tile 15
Time 1: Core[0,0] → tile 16, Core[0,1] → tile 17, ... Core[3,3] → tile 31
Total time: 2 units  ← 16× speedup!
```

---

### **6. Complex 4D DMA for Multi-Column Access (Lines 470-493)**

**Single Core C output:**
```python
# Simple: one column, process sequentially
npu_dma_memcpy_nd(
    sizes=[num_tile_rows, N // n, m, n],     # [2, 8, 64, 32]
    strides=[m * N, n, N, 1],                # [16384, 32, 256, 1]
)
```

**Whole Array C output (4 columns):**
```python
# Complex: INTERLEAVED column access
C_sizes = [
    tb_n_rows,                  # 2 tile rows
    N // n // n_aie_cols,       # 8/4 = 2 tiles per column (DIVIDED!)
    m * n_aie_rows,             # 64*4 = 256 rows per tile (JOINED!)
    n,                          # 32 cols per tile
]
C_strides = [
    m * n_aie_rows * N,         # 256*256 = Next tile row
    n * n_aie_cols,             # 32*4 = Skip to next tile in same column
    N,                          # 256 = Next row
    1,                          # Next element
]
```

**What this does (for column 0):**

```
C matrix (256×256), column 0 shim accesses:

Tile 0: C[0:256, 0:32]      ← First 32 columns
Tile 1: C[0:256, 128:160]   ← Skip 3*32=96 cols (other shims' tiles)
Tile 2: C[0:256, 256:288]   ← Skip another 96 cols
...

Memory pattern:
[row 0: c00...c031, skip 96, c0128...c0159, skip 96, ...]
[row 1: c10...c131, skip 96, c1128...c1159, skip 96, ...]
...
```

**Stride breakdown:**
- `n * n_aie_cols` = 32 * 4 = 128: After reading 32 elements, skip 128 to next tile handled by this shim
- This creates an **interleaved** access pattern where each shim handles every 4th tile column

---

### **7. Variable Column Support (Lines 50, 111, 184-196)**

**Single Core:** Always uses column 0

**Whole Array:**
```python
--n-aie-cols {1, 2, 4, 8}  # Command-line argument

# NPU1 (Phoenix/Hawk):  Max 4 columns (4×4 array)
# NPU2 (Strix/Krackan): Max 8 columns (4×8 array)

if n_aie_cols > n_aie_rows:  # e.g., 8 cols, 4 rows
    n_shim_mem_A = n_aie_rows  # Only use 4 shim/mem tiles for A
    # Alternate columns: use tiles (0,x), (2,x), (4,x), (6,x)
else:
    n_shim_mem_A = n_aie_cols  # Use all available
```

**8-column configuration (NPU2):**

```
Shim tiles used for A:
  shim_tiles[0] ← Col 0
  shim_tiles[2] ← Col 2  (skip col 1!)
  shim_tiles[4] ← Col 4  (skip col 3!)
  shim_tiles[6] ← Col 6  (skip col 5!)

Why skip? We only have 4 row blocks of A, but 8 columns of B.
Distribute A rows across alternating columns to balance load.
```

Code (line 255-257):
```python
shim_tiles[2 * i] if n_aie_cols == 8 else shim_tiles[i]
#         ^^^^^ Use even-numbered columns
```

---

## **Part 3: Data Movement Patterns**

### **Matrix A - Broadcast Pattern**

**Single Core:** No broadcast, just one consumer

**Whole Array:**

```
A[256, 256] split into row blocks:

Block 0: A[0:64, :]    → Broadcast to cores [0,0], [0,1], [0,2], [0,3]
Block 1: A[64:128, :]  → Broadcast to cores [1,0], [1,1], [1,2], [1,3]
Block 2: A[128:192, :] → Broadcast to cores [2,0], [2,1], [2,2], [2,3]
Block 3: A[192:256, :] → Broadcast to cores [3,0], [3,1], [3,2], [3,3]

Each core in a ROW gets the SAME A data!
```

**4D DMA for A (line 535-542):**
```python
A_sizes = [
    N // n // n_aie_cols,       # Reuse count (e.g., 8/4 = 2 times)
    K // k,                      # K tiles (e.g., 4)
    m * n_A_tiles_per_shim,      # Rows per transfer
    k,                           # Cols per K-tile
]
A_strides = [0, k, K, 1]
#           ^ stride = 0 means REUSE same data!
```

---

### **Matrix B - Distribute Pattern**

**Single Core:** Sequential access to all B

**Whole Array:**

```
B[256, 256] split into column blocks:

Block 0: B[:, 0:32]    → All cores [x, 0] (column 0)
Block 1: B[:, 32:64]   → All cores [x, 1] (column 1)
Block 2: B[:, 64:96]   → All cores [x, 2] (column 2)
Block 3: B[:, 96:128]  → All cores [x, 3] (column 3)

Each core in a COLUMN gets the SAME B data!
```

**4D DMA for B (line 586-590):**
```python
B_sizes = [
    N // n // n_aie_cols,    # Tiles per column (8/4 = 2)
    K // k,                  # K tiles
    k, n                     # Tile dimensions
]
B_strides = [
    n * n_aie_cols,          # Skip to next tile in this column
    k * N,                   # Next K-tile
    N, 1                     # Row, element
]
```

---

### **Matrix C - Join Pattern**

**Single Core:** Direct write from one core

**Whole Array:**

```
Each core produces partial C:

Core[0,0] → C[0:64, 0:32]
Core[0,1] → C[0:64, 32:64]
...
Core[3,3] → C[192:256, 96:128]

Join in memory tile, then write to DRAM with interleaved pattern
```

---

## **Part 4: Performance Scaling**

### **Theoretical Speedup**

| Config | Cores | Speedup | Use Case |
|--------|-------|---------|----------|
| Single core | 1 | 1× | Debugging, learning |
| 1 column | 4 (4×1) | 4× | Small problems |
| 2 columns | 8 (4×2) | 8× | Medium problems |
| 4 columns | 16 (4×4) | 16× | Large problems (NPU1 max) |
| 8 columns | 32 (4×8) | 32× | Very large (NPU2 only) |

**Real-world speedup:** ~80-90% of theoretical due to:
- Memory bandwidth limits
- Synchronization overhead
- DMA setup time

---

## **Part 5: Code Complexity Comparison**

### **ObjectFIFO Count**

**Single Core:**
```
inA, memA         (2)
inB, memB         (2)
memC, outC        (2)
Total: 6 FIFOs
```

**Whole Array (4×4):**
```
A: 4 L3→L2 + 4 L2→L1              = 8
B: 4 L3→L2 + 4 L2→L1              = 8
C: 16 L1→L2 (4×4) + 4 L2→L3       = 20
Total: 36 FIFOs (6× more complex!)
```

---

### **Core Loop Complexity**

**Single Core (lines 250-265):**
```python
for _ in range_(0xFFFFFFFF):              # Infinite loop
    for _ in range_(tiles):               # 32 tiles
        elem_out = memC.acquire(...)
        zero(elem_out)
        for _ in range_(K_div_k):         # 4 K-tiles
            elem_in_a = memA.acquire(...)
            elem_in_b = memB.acquire(...)
            matmul(...)
        memC.release(...)
```

**Whole Array (lines 401-425):**
```python
# Nested in row/col loops!
for row in range(n_aie_rows):             # 4 rows
    for col in range(n_aie_cols):         # 4 cols
        @core(core_tiles[row][col], ...):
            for _ in range_(0xFFFFFFFF):
                for _ in range_(n_tiles_per_core):  # Only 2 tiles!
                    elem_out = C_l1l2_fifos[row][col].acquire(...)
                    zero(elem_out)
                    for _ in range_(K // k):
                        elem_in_a = A_l2l1_fifos[row].acquire(...)
                        elem_in_b = B_l2l1_fifos[col].acquire(...)
                        matmul(...)
```

**Key difference:** Each core does LESS work (2 tiles vs 32), but ALL cores run in PARALLEL!

---

## **Part 6: When to Use Which?**

### **Use Single Core When:**
- ✅ Learning MLIR-AIE
- ✅ Debugging new kernels
- ✅ Matrix size < 128×128
- ✅ Rapid prototyping
- ✅ Limited memory (single tile's L1)

### **Use Whole Array When:**
- ✅ Production deployment
- ✅ Matrix size > 512×512
- ✅ Need maximum performance
- ✅ Batch processing
- ✅ Real-time constraints

---

## **Summary Table**

| Feature | Single Core | Whole Array |
|---------|-------------|-------------|
| **Lines of code** | 371 | 627 |
| **Tiles used** | 3 (shim, mem, compute) | 18+ (1 shim + 1 mem + 16 compute) |
| **ObjectFIFOs** | 6 | 36+ |
| **Broadcast** | ❌ No | ✅ Yes (A and B) |
| **Distribute** | ❌ No | ✅ Yes (A into rows) |
| **Join** | ❌ No | ✅ Yes (C from columns) |
| **Tile selection** | `tile(0, 2)` | `tiles[row][col]` |
| **Parallelism** | None | 4-32× |
| **Best for** | Learning | Production |

---

## **Key Takeaways**

1. **Broadcast/Distribute/Join** are the core multi-core patterns
2. **ObjectFIFO linking** with offsets enables data splitting/merging
3. **2D tile arrays** make code scalable
4. **Per-core workload** decreases as core count increases
5. **Complexity cost** is significant but worth it for 16-32× speedup

---

Would you like me to:
1. Explain broadcast/distribute/join in more detail?
2. Show how to scale from 1→4→16 cores step by step?
3. Walk through the exact memory access pattern for a small example (e.g., 128×128)?
