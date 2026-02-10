# CUDA-Kernel-Cases-of-Optimization-Techniques-for-GPU-Programming

A compact set of 28 CUDA kernel examples that demonstrate common optimization techniques in GPU programming. Each `.cu` file is self‑contained and compares a baseline approach against an optimized variant.

## Contents

| File | Optimization Technique | Description |
|---|---|---|
| 01_use_specialized_memory.cu | Specialized memory (constant/shared) | Uses constant memory for hot coefficients to reduce global memory latency. |
| 02_use_warp_primitives.cu | Warp primitives | Uses warp shuffle to reduce shared memory traffic and synchronization in reduction. |
| 03_register_blocking.cu | Register blocking | Computes multiple outputs per thread to reuse data in registers. |
| 04_reduce_register_usage.cu | Reduce register pressure | Simplifies temporaries to improve occupancy and reduce spills. |
| 05_recompute_instead_of_memory.cu | Recompute vs. memory | Recomputes indices to avoid extra global memory loads. |
| 06_coalesced_memory_access.cu | Coalesced access | Converts AoS to SoA layout so warps read contiguous addresses. |
| 07_spatial_tiling.cu | Spatial tiling | Tiles data into shared memory to improve reuse. |
| 08_kernel_fusion.cu | Kernel fusion | Fuses two kernels to reduce launch overhead and intermediate stores. |
| 09_prefetching.cu | Prefetching | Loads data into shared memory early to hide latency. |
| 10_data_compression.cu | Data compression | Packs data to reduce bandwidth and storage costs. |
| 11_precomputation.cu | Precomputation | Uses a lookup table to avoid repeated expensive math. |
| 12_loop_unrolling.cu | Loop unrolling | Unrolls loops to reduce branch/loop overhead. |
| 13_reduce_branch_divergence.cu | Reduce divergence | Uses predication/masking to keep warps on the same control path. |
| 14_sparse_matrix_format.cu | Sparse format selection | Uses CSR instead of COO to improve locality and parallelism. |
| 15_kernel_splitting.cu | Kernel splitting | Splits a heavy kernel into stages to reduce resource pressure. |
| 16_reduce_redundant_work.cu | Reduce redundant work | Avoids repeated loads/computation via shared reuse. |
| 17_vectorization.cu | Vectorization | Uses `float4` loads/stores to process multiple elements per thread. |
| 18_fast_math.cu | Fast math | Uses approximate/fast intrinsics (e.g., `__expf`). |
| 19_warp_centric_programming.cu | Warp-centric programming | Organizes work at warp granularity to reduce sync/communication cost. |
| 20_variable_work_per_thread.cu | Variable work per thread | Uses grid‑stride loops to balance work and reduce tail effects. |
| 21_tune_block_size.cu | Block size tuning | Uses occupancy APIs to pick a better block size. |
| 22_auto_tuning.cu | Auto‑tuning | Searches candidate block sizes to find the fastest. |
| 23_load_balancing.cu | Load balancing | Uses a global counter to dynamically distribute uneven work. |
| 24_reduce_synchronization.cu | Reduce synchronization | Uses warp‑level reduction to cut down `__syncthreads()`. |
| 25_reduce_atomic_ops.cu | Reduce atomics | Uses block‑local histograms before global merge. |
| 26_inter_block_synchronization.cu | Inter‑block sync | Uses cooperative groups for grid‑wide sync (with fallback). |
| 27_host_device_comm_optimization.cu | Host‑device comm | Uses pinned memory + async copies + streams. |
| 28_cpu_gpu_cooperation.cu | CPU/GPU cooperation | Splits workload between CPU and GPU to exploit both. |

## Build & Run

Compile a single file:

```bash
nvcc -O3 -std=c++14 01_use_specialized_memory.cu -o 01_use_specialized_memory
./01_use_specialized_memory
```

Compile all files:

```bash
mkdir -p bin
for f in *.cu; do nvcc -O3 -std=c++14 "$f" -o "bin/${f%.cu}"; done
```

Benchmark all cases with warmup + averaged timing (generates `results.txt`):

```bash
./bench.py --warmup 3 --repeat 10
```

## Benchmark Results (A10, CUDA 12.6)

The following results are **averages over 10 runs** with **3 warmup runs** per case. See `results.txt` for the raw output used below.

```
# Benchmark results (average over 10 runs, 3 warmup runs)
=== 01_use_specialized_memory
max diff: 0.0
global kernel: 0.040 ms, constant kernel: 0.030 ms

=== 02_use_warp_primitives
max diff: 0.0
shared reduce: 0.128 ms, warp reduce: 0.015 ms

=== 03_register_blocking
max diff: 0.0
1 output/thread: 0.120 ms, 2 outputs/thread: 0.029 ms

=== 04_reduce_register_usage
max diff: 0.0
heavy temps: 0.133 ms, light temps: 0.036 ms

=== 05_recompute_instead_of_memory
max diff: 0.0
index array: 0.148 ms, recompute: 0.030 ms

=== 06_coalesced_memory_access
max diff: 0.0
AoS: 0.211 ms, SoA: 0.090 ms

=== 07_spatial_tiling
max diff: 0.0
naive: 0.154 ms, tile: 0.029 ms

=== 08_kernel_fusion
max diff: 1.19209e-07
2 kernels: 0.145 ms, fused: 0.027 ms

=== 09_prefetching
max diff: 0.0
direct: 0.122 ms, shared prefetch: 0.027 ms

=== 10_data_compression
max diff: 1.19209e-07
float: 0.126 ms, packed8: 0.024 ms

=== 11_precomputation
max diff: 0.000972781
sinf: 0.026 ms, LUT: 0.036 ms

=== 12_loop_unrolling
max diff: 0.0
no unroll: 0.152 ms, unroll: 0.023 ms

=== 13_reduce_branch_divergence
max diff: 0.0
branch: 0.120 ms, mask: 0.028 ms

=== 14_sparse_matrix_format
max diff: 1.19209e-07
COO: 0.150 ms, CSR: 0.021 ms

=== 15_kernel_splitting
max diff: 0.0
single kernel: 0.152 ms, split: 0.054 ms

=== 16_reduce_redundant_work
max diff: 0.0
redundant: 0.231 ms, shared: 0.034 ms

=== 17_vectorization
max diff: 0.0
scalar: 0.151 ms, float4: 0.036 ms

=== 18_fast_math
max diff: 2.38419e-07
expf: 0.116 ms, __expf: 0.027 ms

=== 19_warp_centric_programming
max diff: 0.0
per-thread: 0.119 ms, warp-centric: 0.029 ms

=== 20_variable_work_per_thread
max diff: 0.0
fixed: 0.393 ms, grid-stride: 0.299 ms

=== 21_tune_block_size
max diff: 0.0
block128: 0.036 ms, occupancy block 768: 0.028 ms

=== 22_auto_tuning
best block: 256, time: 0.026 ms

=== 23_load_balancing
max diff: 0.0
static: 0.258 ms, dynamic: 0.071 ms

=== 24_reduce_synchronization
max diff: 0.0
shared sync: 0.118 ms, warp shuffle: 0.016 ms

=== 25_reduce_atomic_ops
max diff: 0.0
global atomics: 0.149 ms, block local: 0.063 ms

=== 26_inter_block_synchronization
max diff: 0.0
multi-kernel: 0.151 ms, time: 0.050 ms

=== 27_host_device_comm_optimization
max diff: 0.0
sync copy: 2.727 ms, async pinned: 0.367 ms

=== 28_cpu_gpu_cooperation
max diff: 1.19209e-07
GPU only: 0.102 ms, CPU+GPU: 2.047 ms
```
