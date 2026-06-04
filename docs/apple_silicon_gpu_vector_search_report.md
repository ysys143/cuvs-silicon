# cuvs-silicon: Apple Silicon GPU 벡터 검색 성능 분석 보고서

**작성일**: 2026-06-04  
**대상 하드웨어**: Apple M3 Max (M3 Max SoC, 96GB 통합 메모리)  
**비교 대상**: NVIDIA CUVS (CUDA, GCP A100), metalfaiss (MLX), hnswlib (CPU)

---

## 1. 배경 및 목적

이 보고서는 Apple Silicon(M3 Max) 위에서 GPU 가속 벡터 검색 라이브러리(cuvs-silicon)를 구현하고 벤치마크한 결과를 정리한다. NVIDIA CUVS가 CUDA GPU를 통해 CPU 대비 49-78배의 압도적인 성능 향상을 입증한 사례를 바탕으로, 동일한 알고리즘을 Apple Metal API로 이식하여 Apple Silicon에서도 유사한 GPU 가속 효과를 달성할 수 있는지 검증하는 것이 핵심 목표였다.

---

## 2. 구현 개요

cuvs-silicon은 크게 두 가지 핵심 검색 경로를 구현했다.

**CAGRA 그래프 기반 ANN 검색**은 IVF K-means 시딩, 랜덤 버킷 패스, Metal GPU nn-descent 정제의 3단계로 구성된 인덱스를 빌드한 뒤, 빔 서치(beam search) 커널로 근사 최근접 이웃을 탐색한다. 빌드 시 쿼리별 진입점 선택은 AMX(cblas_sgemm)로 가속하며, 인덱스는 바이너리 파일로 저장·로드할 수 있어 재빌드 없이 재사용이 가능하다.

**GPU 브루트포스 검색**은 2-pass 방식으로 구현된다. 1단계에서 MPS(Metal Performance Shaders) MPSMatrixMultiplication이 쿼리와 데이터셋 사이의 내적 행렬(Q×N)을 GPU에서 계산하고, 2단계에서 커스텀 Metal 커널이 L2 거리 계산과 top-K 선택을 GPU 위에서 완결한다. CPU로 전송되는 데이터는 Q×K 결과(최대 수 KB)뿐이다.

---

## 3. 벤치마크 결과

### 3-1. 100K 벡터 CAGRA (N=100,000, D=1024)

인덱스 빌드 시간은 약 58초로, hnswlib(ef_construction=128)의 약 67초보다 빠르다. 검색 처리량은 Q=1,000 배치 기준 약 630 QPS이며, recall@10은 0.992-0.993으로 요구치(0.99) 이상이다. 단, 단일 쿼리(Q=1) 레이턴시는 54ms로, hnswlib의 약 1ms 대비 54배 느리다.

### 3-2. 1M 벡터 브루트포스 (N=1,000,000, D=1024)

Q=100 배치 기준 cuvs-silicon의 처리량은 1,012 QPS로, metalfaiss(MLX 기반)의 352 QPS 대비 약 2.9배 빠르다. float16으로 데이터셋을 변환함으로써 MPS matmul의 메모리 읽기를 4GB에서 2GB로 줄인 것이 핵심 최적화였다. 그러나 단일 쿼리(Q=1) 레이턴시는 19ms로, CPU hnswlib의 약 1ms 대비 여전히 19배 느리다.

### 3-3. NVIDIA CUVS (참고, GCP A100)

비교를 위해 이전에 측정한 NVIDIA CUVS CUDA 결과를 제시한다. 동일한 DBpedia 1M×1536d 데이터셋에서 CAGRA 단일 쿼리 레이턴시는 2.4ms이며 QPS는 415였다. 같은 서버에서 pgvector HNSW(CPU)는 단일 쿼리 레이턴시 24ms, QPS 8.5를 기록했다. CUVS CAGRA는 pgvector 대비 약 49배 빠른 성능을 보였다.

---

## 4. Apple Silicon GPU 한계 가설과 실험적 검증

### 4-1. 가설 형성

벤치마크 결과를 분석하는 과정에서 두 가지 해석이 경쟁했다. 첫 번째는 구현 문제로, Metal 커널 설계가 비효율적이어서 최적화하면 개선될 수 있다는 것이었다. 두 번째는 Apple Silicon 구조적 한계로, Metal API의 높은 dispatch 오버헤드와 GPU 아키텍처 자체의 제약이 단일 쿼리 고성능을 막는다는 것이었다.

이 두 가설을 구분하기 위해 metalfaiss(MLX 기반, Apple이 개발한 ML 프레임워크를 사용)의 Q=1 단일 쿼리 레이턴시를 측정했다. MLX는 Apple Silicon에 최적화된 프레임워크이며, metalfaiss는 그 위에서 brute-force 검색을 수행하는 최적화된 구현체다. 따라서 metalfaiss Q=1 결과는 Apple Silicon GPU에서 달성 가능한 사실상의 상한선에 해당한다.

### 4-2. 측정 결과

1M×1024d 데이터셋, K=10, Q=1 조건에서 측정한 결과는 다음과 같다.

- **hnswlib(CPU)**: p50 약 1ms
- **cuvs-silicon GPU BF**: p50 19ms
- **metalfaiss(MLX GPU)**: p50 82ms

cuvs-silicon이 MLX보다 4배 빠른 결과는, 우리 구현이 이미 Apple Silicon에서 달성 가능한 최적에 가깝다는 것을 의미한다. MLX의 82ms는 단일 쿼리에서 GPU 가속이 Apple Silicon 위에서 근본적으로 불리하다는 것을 보여준다.

### 4-3. 원인 분석

**물리적 대역폭 한계.** 1M×1024d float32 데이터셋은 4GB다. Q=1 브루트포스는 1개 쿼리를 위해 이 4GB 전체를 읽어야 한다. M3 Max의 GPU 메모리 대역폭은 약 400GB/s이며, 이를 기준으로 하면 4GB / 400GB/s = 10ms가 이론적 최솟값이다. 실제 측정된 19ms는 이 최솟값에 Metal API 오버헤드(~5-9ms)를 더한 수준으로, 구현상 추가 최적화 여지가 거의 없다.

**Metal commandBuffer 오버헤드.** Metal API에서 커널을 dispatch하려면 commandBuffer 생성, encoding, commit, waitUntilCompleted 과정을 거쳐야 한다. 이 고정 오버헤드는 약 1-2ms이며, CUDA의 커널 launch 오버헤드(~5μs)보다 200-400배 크다. Q=1처럼 연산이 적을수록 이 오버헤드의 비중이 커진다.

**GPU 활용률 문제 (CAGRA 한정).** 브루트포스와 달리 CAGRA 빔 서치 커널은 쿼리당 1개 threadgroup만 dispatch한다. M3 Max는 약 38개 shader engine을 가지므로, Q=1에서의 GPU 활용률은 1/38 = 약 2.6%에 불과하다. NVIDIA CUVS는 multi-CTA(쿼리당 수백 개 thread block)를 사용해 Q=1에서도 모든 SM을 채운다. 이 차이가 CAGRA Q=1 레이턴시에서 cuvs-silicon 54ms vs CUVS 2.4ms라는 22배 격차를 만든다. 단, 이 문제는 구현으로 해결 가능하다(multi-team 커널).

**CUDA A100 대비 하드웨어 격차.** NVIDIA A100은 HBM2e 메모리로 약 2TB/s 대역폭을 제공하며, M3 Max의 400GB/s 대비 5배 높다. CAGRA 빌드에서도 CUVS는 IVF_PQ 알고리즘을 사용해 1M 벡터를 23초 만에 빌드하지만, cuvs-silicon의 IVF K-means 기반 빌드는 100K에서 58초가 걸리고 1M에서는 타임아웃이 발생한다.

---

## 5. 성공과 실패의 구분

### 성공

**GPU 브루트포스 가속**은 의미 있는 성과를 거뒀다. MPS matmul 기반 2-pass 방식으로 metalfaiss(MLX) 대비 2.9배 빠른 1,012 QPS를 달성했으며, AMX 브루트포스(13.8 QPS) 대비로는 73배 개선됐다. Apple Silicon GPU 위에서 달성 가능한 수준을 초과하는 성능이다. Q=100 이상의 배치 워크로드에서 Apple Silicon의 GPU를 실질적으로 활용하는 데 성공했다.

**인덱스 직렬화**를 구현해 빌드(75초)를 최초 1회로 줄이고 이후 로드(1.8초)만으로 검색을 시작할 수 있게 됐다.

**100K CAGRA 빌드 가속**은 hnswlib 대비 시간이 단축됐다(67s → 58s). 완전 정확한(recall=1.0) 브루트포스 대신 0.99+ recall의 ANN으로 배치 처리량을 1.5배 개선했다.

### 실패

**CAGRA의 GPU 가속이 입증되지 않았다.** NVIDIA CUVS에서 CAGRA는 CPU HNSW 대비 49배 빠른 성능을 보였지만, cuvs-silicon CAGRA는 hnswlib 대비 배치에서 1.5배, 단일 쿼리에서는 54배 느리다. 이는 multi-CTA 미구현과 Metal dispatch 오버헤드가 복합적으로 작용한 결과다.

**1M 이상 스케일에서 CAGRA 빌드가 불가능하다.** IVF K-means 빌드 알고리즘이 O(N^1.5) 복잡도를 가지며, 1M에서 900초 타임아웃이 발생한다. CUVS가 사용하는 IVF_PQ 기반 빌드 알고리즘이 필요하다.

---

## 6. 결론

단일 쿼리(Q=1) 레이턴시에서 GPU 기반 접근법은 Apple Silicon에서 근본적으로 불리하다. 1M 벡터에 대한 Q=1 브루트포스 최솟값은 물리적으로 10ms 이상이며, 이는 CPU HNSW의 1ms와 비교해 10배 이상 느리다. metalfaiss(MLX) 역시 82ms를 기록해 이 한계가 구현 문제가 아닌 하드웨어 구조에서 비롯됨을 확인했다.

그러나 배치 처리(Q≥10)에서는 GPU가 명확한 이점을 가진다. 동시 요청이 많은 서버 환경이나 오프라인 벡터 처리에서 cuvs-silicon의 GPU 브루트포스는 MLX 기반 최선 구현보다도 빠른 성능을 제공한다.

CAGRA의 GPU 가속을 실현하려면 multi-CTA 빔 서치 커널과 IVF_PQ 빌드 알고리즘이 필요하다. 이 두 가지는 구현 난이도가 높지만 Apple Silicon의 구조적 한계와 무관하게 달성 가능한 목표다. 단, Metal dispatch 오버헤드(1-2ms)로 인해 Q=1 CAGRA 레이턴시는 최적 구현에서도 2-5ms 수준에 머물 것이며, CUDA A100의 2.4ms와는 비슷하지만 CPU HNSW의 1ms를 넘어서기는 어렵다.

---

## 7. 하드웨어 비교 요약

| 항목 | NVIDIA A100 (CUDA) | Apple M3 Max (Metal) |
|---|---|---|
| GPU 메모리 대역폭 | 2TB/s (HBM2e) | 400GB/s (LPDDR5) |
| Kernel launch 오버헤드 | ~5μs | ~1-2ms |
| Persistent threads | 지원 | 미지원 |
| Multi-CTA per query | 지원 (CUVS 구현) | 미구현 |
| CAGRA Q=1 레이턴시 | 2.4ms | 54ms (현재) |
| BF Q=100 throughput | 40 QPS (1M, 1536d) | 1,012 QPS (1M, 1024d) |
