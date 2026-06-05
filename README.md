# cuvs-silicon

[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Metal](https://img.shields.io/badge/Metal-3.0+-silver.svg)](https://developer.apple.com/metal/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3-black.svg)](https://www.apple.com/mac/)
[![Apache 2.0 License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

GPU-accelerated vector similarity search for Apple Silicon, built on Metal Compute Shaders and the Accelerate framework (AMX). Implements brute-force exact search and CAGRA graph-based approximate nearest neighbor (ANN) search.

---

## Highlights

- **Brute-force exact search**: 2.87× faster than metalfaiss (MLX) at N=1M, D=1024
- **CAGRA ANN**: beats hnswlib build time at N=100K with competitive recall
- **2-pass GPU pipeline**: MPS float16 matmul + GPU top-K kernel — only Q×K results leave the GPU
- **Index persistence**: save/load built CAGRA index to skip rebuild

---

## Benchmarks

### Brute-Force Search — N=1M, D=1024, K=10 (M3 Max, Q=100)

| Implementation | QPS | p50 latency | recall@10 |
|---|---|---|---|
| **cuvs-silicon** | **1012** | **99 ms** | 1.0000 |
| metalfaiss (MLX) | 351 | 275 ms | 1.0000 |
| AMX cblas_sgemm (CPU) | 14 | — | 1.0000 |

### CAGRA ANN — N=100K, D=1024 (M3 Max)

| | cuvs-silicon | hnswlib ef=128 |
|---|---|---|
| Build time | 149 s | 67 s |
| Search QPS | 518 | ~430 |
| recall@10 | 0.9960 | ~0.97 |
| Q=1 latency | 54 ms | ~1 ms |

> Single-query (Q=1) latency is hardware-limited on Apple Silicon. Any GPU approach — including metalfaiss (82 ms at Q=1) — is slower than CPU HNSW (~1 ms) for single queries due to Metal command buffer overhead (~2 ms fixed cost) and unified memory bandwidth constraints. See [docs/apple_silicon_gpu_vector_search_report.md](docs/apple_silicon_gpu_vector_search_report.md).

---

## Installation

**Requirements**: macOS 14+, Xcode 15+, Apple Silicon (M1/M2/M3), CMake 3.20+

```bash
git clone https://github.com/ysys143/cuvs-silicon.git
cd cuvs-silicon
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

---

## Usage

### Brute-Force Exact Search

Pass `graph_degree=0` to skip graph build and use exact GPU search:

```bash
./build/cohere_wiki_validation \
    data/1m/base.bin data/1m/queries.bin data/1m/gt.bin \
    10 1 3 0   # K warmup measure graph_degree=0
```

### CAGRA ANN Search

```bash
# Build index and search
./build/cohere_wiki_validation \
    data/100k/base.bin data/100k/queries.bin data/100k/gt.bin \
    10 1 3     # K warmup measure [graph_degree=64]

# Save index after build  (argv[8] = save path)
./build/cohere_wiki_validation ... 10 1 3 64 /path/to/index.idx

# Load saved index and search  (argv[9] = load path)
./build/cohere_wiki_validation ... 10 1 3 64 "" /path/to/index.idx
```

### Download Benchmark Data

```bash
python3 scripts/download_cohere_wiki.py   # downloads 10K / 100K / 1M splits
```

---

## Architecture

### Brute-Force (2-pass GPU)

```
Pass 1  MPS MPSMatrixMultiplication (float16 inputs)
        queries[Q×D] @ dataset[N×D]^T  →  cross[Q×N]   (GPU-private buffer)

Pass 2  Metal kernel: l2_topk_from_cross
        cross + q_norms + d_norms  →  top-K indices/distances
        CPU receives only Q×K bytes (e.g. 100 queries × 10 results × 8 B = 8 KB)
```

Dataset and d_norms are cached across calls as float16 Metal buffers.

### CAGRA Build Pipeline

```
Phase 1a  IVF K-means seeding
          N ≤ 200K : exact cblas_sgemm per cluster (AMX, L3-resident)
          N > 200K : IVF_PQ — LUT distance lookup, CPU PQ training

Phase 1b  GPU random bucketing  (Metal kernel, cross-cluster diversity)

Phase 2   nn-descent refinement (Metal kernel, float4 SIMD distances)
```

Entry-point selection at search time uses AMX-accelerated cblas_sgemm over cached navigation vectors.

---

## CAGRA Limitations

### Single-query latency

CAGRA Q=1 latency is 54 ms on cuvs-silicon vs. 2.4 ms on NVIDIA CUVS (A100).  
Root cause: Metal command buffer fixed overhead (~2 ms) + single threadgroup dispatched for Q=1 → 2.6% GPU utilization. Multi-CTA beam search was implemented and tested but worsened latency due to cache thrashing on memory-bound traversal.

### 1M-scale build

| Stage | Time | Root cause |
|---|---|---|
| K-means (chunk=2048) | 341 s | M-step scatter-add is DRAM-bound on CPU |
| PQ training | 764 s | 8 subspace × N=1M → 229 B memory ops |
| CPU PQ seeding | 1548 s | 229 B pair lookups × ~3 ns each |
| GPU bucketing + misc | ~60 s | — |
| **Total** | **2714 s** | — |

NVIDIA CUVS completes the same workload in 22 s (A100, HBM2e 2 TB/s vs. CPU ~50 GB/s effective).  
A Metal GPU kernel for the PQ seeding step would reduce the bottleneck to ~100 s.  
See [docs/multi-cta-ivf-pq-lessons.md](docs/multi-cta-ivf-pq-lessons.md) for full analysis.

---

## Contributing

Contributions are welcome. The highest-impact areas:

| Area | Expected impact | Status |
|---|---|---|
| **Metal GPU K-means M-step** (scatter-add kernel) | 1M build: 341 s → ~50 s | open |
| **Metal GPU IVF_PQ seeding kernel** | 1M build: 1548 s → ~100 s | open — [draft kernel exists](shaders/cagra_kernels.metal), bug in PQ distance scale |
| **nn-descent optimization for N=1M** | enables recall improvement at 1M | open |
| Multi-CTA beam search (Q=1 latency) | 54 ms → ~10 ms | open — [kernel implemented](shaders/cagra_kernels.metal), disabled |

Please open an issue before starting large changes.

---

## Related Projects

- [NVIDIA CUVS](https://github.com/rapidsai/cuvs) — CUDA reference implementation
- [MetalFaiss / Faiss-mlx](https://github.com/MLXPorts/Faiss-mlx) — MLX-based FAISS (comparison baseline)
- [hnswlib](https://github.com/nmslib/hnswlib) — CPU HNSW baseline

---

## License

Apache 2.0 — Copyright 2026 JAESOL SHIN
