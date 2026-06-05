# cuvs-silicon

Apple Silicon(M-series) Metal GPU를 활용한 벡터 유사도 검색 라이브러리.  
[NVIDIA CUVS](https://github.com/rapidsai/cuvs)의 핵심 알고리즘(CAGRA, Brute-Force)을  
Metal Compute Shader로 이식하고, Apple의 AMX/MPS/MLX 가속을 함께 활용한다.

---

## 주요 결과

### Brute-Force 검색 (GPU 정확 탐색)

N=1M, D=1024, K=10, Q=100 쿼리 기준 (M3 Max):

| 구현 | QPS | p50 레이턴시(Q=100) | recall@10 |
|---|---|---|---|
| **cuvs-silicon** | **1012** | **99ms** | 1.0000 |
| metalfaiss (MLX) | 351 | 275ms | 1.0000 |
| AMX cblas_sgemm | 13.8 | - | 1.0000 |

metalfaiss 대비 **2.87× 빠름**.  
2-pass 구조: MPS float16 행렬곱(dataset 한 번 읽기) + GPU top-K 커널(Q×K만 CPU로 전송).

### CAGRA 그래프 기반 ANN 검색

N=100K, D=1024 (M3 Max):

| 항목 | cuvs-silicon | hnswlib |
|---|---|---|
| 빌드 시간 | 149s | 67s |
| search QPS | 518 | ~430 |
| recall@10 | 0.9960 | ~0.97 |
| Q=1 레이턴시 | 54ms | ~1ms |

N=1M:
- 빌드: 2714s (완료), recall=0.72 (Q=1 레이턴시 미측정)
- CUDA CUVS 비교: 22s 빌드, recall=0.986 (A100 40GB)

---

## 아키텍처

### Brute-Force (2-pass GPU)

```
[Pass 1] MPS MPSMatrixMultiplication (float16)
  queries[Q×D] @ dataset[N×D]^T → cross[Q×N]  (GPU 전용 버퍼)

[Pass 2] Metal 커널 l2_topk_from_cross
  cross[Q×N] + q_norms + d_norms → top-K indices/distances  (Q×K만 CPU로)
```

- dataset float16 변환 캐시: N×D×2 bytes (1M×1024: 2GB)
- GPU 버퍼 캐시: d_norms, dataset_fp16 반복 호출 시 재사용

### CAGRA (그래프 기반 ANN)

```
빌드:
  Phase 1a: IVF K-means 세딩 (N≤200K: exact sgemm, N>200K: IVF_PQ)
  Phase 1b: random bucketing (Metal GPU 커널, cross-cluster 연결)
  Phase 2:  nn-descent 정제 (Metal GPU 커널, float4 SIMD)

검색:
  AMX cblas_sgemm으로 nav_vectors에서 진입점 선택
  Metal GPU beam search (beam_size=32~512, float4 거리)
```

---

## CAGRA 한계와 원인 분석

NVIDIA CUVS CAGRA는 Q=1 레이턴시 2.4ms, 22s 빌드(1M×1536d)를 달성한다.  
cuvs-silicon이 이를 재현하지 못하는 구조적 이유:

### 1. 단일 쿼리 레이턴시 (Q=1)

cuvs-silicon: **54ms** / hnswlib CPU: **~1ms** / CUVS CUDA: **2.4ms**

```
원인:
- Metal commandBuffer dispatch 오버헤드: ~2ms (CUDA: ~5μs)
- Q=1 = 1 threadgroup → GPU 2.6% 활용 (CUDA multi-CTA: 전 SM 포화)
- 1M 데이터셋 random access 최솟값: 4GB/400GB·s = 10ms
```

metalfaiss(MLX)도 Q=1에서 82ms로 cuvs-silicon(19ms, brute-force)보다 느림.  
→ **Q=1 고레이턴시는 Apple Silicon GPU의 구조적 한계**임을 실험으로 확인.

### 2. 1M 빌드 시간

cuvs-silicon: **2714s** / CUVS CUDA: **22s**

```
병목 분석 (타임스탬프 실측):
  K-means (chunk=2048): 341s  ← M-step scatter-add DRAM bottleneck
  PQ training (8 subspaces): 764s  ← 229B lookup × ~3ns = too slow on CPU
  CPU PQ seeding: 1548s         ← 229B pair × 8 LUT reads × CPU bandwidth
  GPU bucketing: ~30s           ← Metal GPU, fast
  Total: 2714s
```

CUDA A100은 HBM2e(2TB/s)로 같은 229B 연산을 ~8s에 처리.  
M3 Max CPU는 DRAM effective bandwidth ~50GB/s → **40× 느림**.

### 3. 무엇이 필요한가

CAGRA의 완전한 GPU 가속을 위해서는:
- K-means M-step: Metal GPU atomic scatter-add 커널
- IVF_PQ 세딩: Metal GPU PQ lookup 커널 (pq_codes + LUT 전부 GPU 메모리에)
- nn-descent: 현재 구현은 1M에서 ~900s/iteration → 최적화 필요

이 세 가지가 모두 Metal GPU로 이식되면 이론적으로 100-200s 빌드가 가능하다.

---

## 빌드

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

요구사항: macOS 14+, Xcode 15+, Apple Silicon (M1/M2/M3)

---

## 벤치마크

```bash
# Cohere Wikipedia 1M×1024d 데이터 다운로드
python3 scripts/download_cohere_wiki.py

# Brute-force (GPU 정확 탐색)
./build/cohere_wiki_validation data/1m/cohere_wiki_base.bin \
    data/1m/cohere_wiki_queries.bin data/1m/cohere_wiki_gt.bin 10 1 3 0

# CAGRA ANN
./build/cohere_wiki_validation data/100k/cohere_wiki_base.bin \
    data/100k/cohere_wiki_queries.bin data/100k/cohere_wiki_gt.bin 10 1 3

# 인덱스 저장/로드
./build/cohere_wiki_validation ... 10 1 3 64 /path/to/index.idx       # 저장
./build/cohere_wiki_validation ... 10 1 3 64 "" /path/to/index.idx    # 로드
```

---

## 기여 환영

다음 영역에서 기여를 환영합니다:

**높은 우선순위**
- [ ] **Metal GPU K-means M-step 커널**: scatter-add를 GPU로 이식하면 1M 빌드 시간이 50% 이상 단축될 것으로 예상
- [ ] **Metal GPU IVF_PQ 세딩 커널**: `pq_cluster_seeding` 커널의 버그 수정 또는 재설계 (현재 PQ distance scale 문제로 비활성화)
- [ ] **1M CAGRA recall 개선**: nn-descent GPU 커널 최적화로 1M에서 nd_iters≥1 달성

**중간 우선순위**
- [ ] Multi-CTA beam search: Q=1 레이턴시를 10ms 이하로 (현재 커널 구현됨, 검증 필요)
- [ ] IVF_PQ 빌드 파이프라인 최적화 (K-means + PQ training 속도)
- [ ] faiss-mlx와의 통합 또는 비교 벤치마크

**낮은 우선순위**
- [ ] ILP64 Accelerate API 마이그레이션 (deprecated cblas 경고 제거)
- [ ] Python 바인딩
- [ ] PostgreSQL 통합 (pg_cuvs 참조)

---

## 관련 프로젝트

- [NVIDIA CUVS](https://github.com/rapidsai/cuvs) — CUDA 기반 원본 구현
- [metalfaiss](https://github.com/) — MLX 기반 FAISS (비교 대상)
- [hnswlib](https://github.com/nmslib/hnswlib) — CPU HNSW 기준선
- [pg_cuvs](../pg_cuvs) — PostgreSQL 확장 (CUDA)

---

## 라이선스

Apache 2.0
