# cuvs-silicon — MVP 실행 명세

> seed_4fd4bb55deea v1.6.0 | ambiguity 0.16 | 2026-05-31

---

## 목표

cuVS CAGRA C++ 헤더 인터페이스를 FAISS 호환성 레이어로 제공하면서, Metal Shading Language(.metal 파일)로 작성된 GPU 커널과 Metal 프레임워크(MTLDevice/MTLCommandQueue/MTLComputePipelineState) 호스트 코드를 사용해 Apple Silicon에서 CAGRA ANN 검색을 가속한다.

SIFT-1M 벤치마크에서 두 시스템이 모두 recall@10 >= 0.90을 달성할 때, cuvs-silicon가 Faiss-mlx 대비 recall@10, QPS, p99 latency 모두에서 우위를 가져야 한다.

---

## 제약 조건

### 구현 필수
- MVP 검색은 Metal GPU에서 실행해야 함 — CPU fallback 불가
- MVP 인덱스 빌드는 CPU-assisted 그래프 구성(hnswlib 등) 허용
- GPU 커널은 `.metal` 파일의 MSL로 작성, `xcrun metal` 또는 CMake Metal 컴파일
- 호스트 코드는 `MTLDevice`, `MTLCommandQueue`, `MTLComputePipelineState`, `MTLBuffer`, `MTLComputeCommandEncoder` 사용 필수
- `CMakeLists.txt`는 `Metal.framework`, `CoreFoundation.framework` 링크, `.metal` 파일 컴파일 포함 — `.metal` 파일 없으면 빌드 실패
- `cagra::search()`는 `MTLComputeCommandEncoder`로 최소 1개 커널 dispatch 필수
- `std::sort`, `std::partial_sort`, CPU 루프만으로 구성된 검색 경로는 위반

### API 표면 (FAISS 호환)
```cpp
// cuvs/neighbors/cagra.hpp
namespace cuvs::neighbors::cagra {
  struct index_params { ... };
  struct search_params { ... };
  template<typename DataT, typename IndexT = uint32_t> class index { ... };

  index<float,uint32_t> build(
    const raft::resources&, const index_params&,
    raft::device_matrix_view<const float, int64_t> dataset);

  void search(
    const raft::resources&, const search_params&,
    const index<float,uint32_t>&,
    raft::device_matrix_view<const float, int64_t> queries,
    raft::device_matrix_view<uint32_t, int64_t> neighbors,
    raft::device_matrix_view<float, int64_t> distances);
}

// raft/core/resources.hpp — header-only shim, RAFT 런타임 링크 불필요
// raft::device_matrix_view — unified-memory pointer+extents, CUDA device pointer 불필요
```

### 금지 의존성
- CUDA runtime
- RAFT runtime library
- MLX (benchmark harness에서 Faiss-mlx 설치는 허용)
- 직렬화 (MVP 범위 외)
- Milvus, Lucene 통합 (follow-on)

### TDD 필수
- 각 AC마다 실패하는 테스트를 먼저 작성 (컴파일 오류는 Red 단계 불인정)
- Metal GPU 경로 테스트는 실제 GPU dispatch를 assert (CPU fallback 결과만으로는 불충분)
- CPU fallback은 Metal 경로 테스트에서 비활성화

---

## 완료 기준 (Acceptance Criteria)

### 레벨 1 — 인프라 (선행 필요)

| # | 기준 | 상태 |
|---|---|---|
| AC 3 | SIFT-1M 128d float32 L2 데이터셋 로더 | FAILED |
| AC 8 | 하드웨어 보고: 칩 모델, GPU 코어 수, 통합 메모리 크기 | FAILED |
| AC 9 | macOS 14+ 검증, Xcode 버전 보고 | FAILED |
| AC 10 | FAISS v1.8.0 태그 호환성 테스트 | FAILED |

### 레벨 2 — Metal GPU 핵심 구현

| # | 기준 | 상태 |
|---|---|---|
| AC 1 | FAISS v1.8.0 `faiss/gpu/cuvs/`가 cuvs-silicon 링크로 컴파일 (CMake 경로만 변경, .cpp/.h 수정 불가) | pending |
| AC 2 | `faiss/gpu/test/TestGpuIndexCagra`가 Apple Silicon에서 CUDA 없이 통과 | pending |
| AC 7 | Metal GPU QPS > CPU HNSW QPS (동일 머신) | pending |

### 레벨 3 — 벤치마크 프로토콜

| # | 기준 | 상태 |
|---|---|---|
| AC 4 | recall@10 >= 0.90 (SIFT-1M, 10,000 쿼리) | pending |
| AC 5 | batch_size=100, p99 per-batch ms, 5 warmup + 10 measured | pending |
| AC 6 | CPU HNSW baseline: M=16, efConstruction=200, efSearch=128, single-threaded | pending |
| AC 13 | 보고 메트릭: QPS, p50/p99 latency, recall@10, build time (정보용) | pending |

### 레벨 4 — 비교 검증

| # | 기준 | 상태 |
|---|---|---|
| AC 11 | Faiss-mlx 벤치마크: IVFFlat nprobe-tuned (또는 Flat), batch_size=100, 동일 하드웨어, 정확한 버전 기록 | pending |
| AC 12 | 두 시스템 모두 recall@10 >= 0.90 시: cuvs-silicon recall >= Faiss-mlx AND QPS > Faiss-mlx AND p99 < Faiss-mlx | pending |

---

## 벤치마크 환경 고정

| 항목 | 값 |
|---|---|
| 데이터셋 | SIFT-1M, 128d, float32, L2 |
| 쿼리 수 | 10,000 |
| Warmup / Measured | 5회 / 10회 평균 |
| Batch size | 100 |
| Latency 단위 | ms/batch |
| CPU HNSW | M=16, efConstruction=200, efSearch=128, single-threaded |
| 하드웨어 | Apple Silicon M-series M1+ (칩 모델, GPU 코어 수, 메모리 기록) |
| OS | macOS 14 Sonoma+, Xcode 버전 기록 |
| FAISS 버전 | v1.8.0 tag |
| Faiss-mlx 버전 | 실행 시점 최신 v0.1.x, 정확한 버전 기록 |

---

## 평가 가중치

| 원칙 | 가중치 |
|---|---|
| API 호환성 (FAISS v1.8.0 컴파일 + 테스트 통과) | 0.25 |
| 검색 정확도 (recall@10 >= 0.90) | 0.20 |
| 성능 vs baseline (Metal > CPU HNSW, Faiss-mlx 능가) | 0.20 |
| 벤치마크 재현성 | 0.15 |
| 아키텍처 적합성 (No CUDA/RAFT/MLX, Metal Compute Shader) | 0.10 |
| TDD 준수 | 0.10 |

---

## 종료 조건

- **Full pass**: TestGpuIndexCagra 통과 + recall@10 >= 0.90 + Metal GPU QPS > CPU HNSW + Faiss-mlx 능가
- **Stop for redesign**: FAISS v1.8.0 cuVS 통합이 Apple Silicon 통합 메모리로 추상화할 수 없는 CUDA 타입을 요구하는 경우

---

## 로드맵

- v1.0: nn-descent 그래프 구성 및 beam search 전체 Pure Metal Compute Shader 구현
- future: Milvus 통합, Apache Lucene 통합, 인덱스 직렬화
