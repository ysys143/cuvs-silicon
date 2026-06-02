# cuvs-silicon — 구현 태스크 목록

> AC 의존성 순서 기반 | 마지막 실행: orch_3b9008c0f1eb (2026-06-01, AC 0/13)

---

## Sub-AC 분해 (Ouroboros 실행에서 추출)

Ouroboros가 13개 AC를 세분화한 테스트 가능 단위 목록.

### AC 3 — SIFT-1M 데이터셋
- [ ] Sub-AC 1: fvecs reader — base/query 벡터 128d float32 로드
- [ ] Sub-AC 2: ivecs reader — ground-truth neighbor 인덱스 로드
- [ ] Sub-AC 3: 통합 로더 — base/query/groundtruth 결합, 차원/개수 검증
- [ ] Sub-AC 4: 1,000,000개 벡터 수 검증, 비SIFT 거부
- [ ] Sub-AC 5: L2 거리 계산 함수, 128d float32 known pair 테스트

### AC 5 — 벤치마크 프로토콜
- [ ] Sub-AC 1: batch_size=100 쿼리/호출 검증
- [ ] Sub-AC 2: warmup 5회 샘플 측정값에서 제외 검증
- [ ] Sub-AC 3: p99 계산 — 10회 per-batch latency 백분위, ms 단위 레이블 포함

### AC 6 — CPU HNSW 베이스라인
- [ ] Sub-AC 1: M=16, efConstruction=200 파라미터 노출 + 테스트
- [ ] Sub-AC 2: efSearch=128 쿼리 시점 적용 + 테스트
- [ ] Sub-AC 3: 인덱싱 single-threaded 강제 + 테스트
- [ ] Sub-AC 4: 검색 single-threaded 강제 + 테스트

### AC 8 — 하드웨어 보고
- [ ] Sub-AC 1: 칩 모델 파싱 (sysctl/system-profiler mock 입력 테스트)
- [ ] Sub-AC 2: GPU 코어 수 파싱 (system-profiler mock 테스트)
- [ ] Sub-AC 3: 통합 메모리 크기 파싱 (byte/human-readable 포맷 테스트)
- [ ] Sub-AC 4: 세 필드 집계 → 벤치마크 출력에 포함 + 직렬화 테스트

### AC 9 — macOS / Xcode 버전
- [ ] Sub-AC 1: macOS 버전 파싱 (comparable version components)
- [ ] Sub-AC 2: macOS 14.0 미만 거부 + 14.0 이상 수락 테스트
- [ ] Sub-AC 3: Xcode 버전 감지 함수
- [ ] Sub-AC 4: 벤치마크 출력에 Xcode 버전 포함 테스트

### AC 10 — FAISS v1.8.0 호환성
- [ ] Sub-AC 1: 헤더 컴파일 테스트 (링크 없이 include + 사용)
- [ ] Sub-AC 2: 링크 테스트 (cuVS CAGRA 심볼 resolve 확인)
- [ ] Sub-AC 3: 런타임 smoke test — create → add → search → 결과 shape/correctness

### AC 2 — TestGpuIndexCagra (FAISS 인터페이스 계약)
- [ ] Sub-AC 1: 검색 결과 shape — single-query/batched 차원 FAISS v1.8.0 호환
- [ ] Sub-AC 2: 검색 결과 정렬 — FAISS v1.8.0 distance ordering 계약 준수
- [ ] Sub-AC 3: 검색 결과 타입 — label/distance 요소 타입 호환
- [ ] Sub-AC 4: 잘못된 입력 처리 — FAISS v1.8.0 에러 처리 매칭

### AC 7 — Metal GPU 검색 (핵심, sub-AC 미분해)
- [ ] `shaders/cagra_search.metal` — MSL beam search 커널
- [ ] `src/metal_search.mm` — MTL 파이프라인 + GPU dispatch
- [ ] Metal dispatch 어서션 테스트 (CPU fallback 비활성화)
- [ ] Metal GPU QPS > CPU HNSW QPS 측정

---

## 우선순위 1 — 인프라 수정 (AC 3, 8, 9, 10 실패 원인 해소)

이 4개가 통과해야 나머지 AC가 실행 가능하다.

### AC 3: SIFT-1M 데이터셋 로더

- [ ] SIFT-1M 데이터셋 다운로드 스크립트 작성 (`scripts/download_sift1m.sh`)
  - base: `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz` (500MB)
  - 또는 ANN-Benchmarks 미러 활용
- [ ] `include/cuvs_silicon/sift_loader.hpp` — fvecs/ivecs 파서 완성
- [ ] `tests/test_sift_loader.cpp` Red → Green 확인
  - Red: 파일 없을 때 명확한 에러
  - Green: base 1M×128, query 10K×128, groundtruth 10K×100 로드

### AC 8: 하드웨어 정보 보고

- [ ] `include/cuvs_silicon/hardware.hpp` — 칩 모델, GPU 코어 수, 통합 메모리 크기 감지
  - `sysctl hw.model` → 칩 모델
  - `system_profiler SPDisplaysDataType` → GPU 코어 수
  - `sysctl hw.memsize` → 통합 메모리
- [ ] `tests/test_hardware_detection.cpp` Red → Green 확인
  - Red: 필드 누락 시 실패
  - Green: 세 필드 모두 populated

### AC 9: macOS / Xcode 버전 보고

- [ ] `sw_vers -productVersion` 파싱 → macOS 14.0 미만 시 preflight 실패
- [ ] `xcodebuild -version` 파싱 → Xcode 버전 문자열 반환
- [ ] `tests/test_hardware_detection.cpp`에 버전 검증 케이스 추가

### AC 10: FAISS v1.8.0 호환성

- [ ] FAISS v1.8.0 설치 또는 `build/faiss_v1_8_cagra_runtime_smoke/faiss-1.8.0/` 빌드
  - `cmake -DFAISS_ENABLE_CUVS=ON -DFAISS_CUVS_LIBRARY=<cuvs-silicon 경로> ...`
- [ ] `tests/faiss_v1_8_cuvs_header_probe.cpp` 컴파일 통과
- [ ] `tests/faiss_v1_8_cagra_runtime_smoke.cpp` 실행 통과

---

## 우선순위 2 — Metal GPU 핵심 (AC 7 → AC 1, 2)

**Metal 구현 없이는 AC 1, 2, 4, 5, 6, 7, 11, 12 모두 통과 불가.**

### Metal 검색 커널 (AC 7의 핵심)

- [ ] `shaders/cagra_search.metal` 작성 (MSL)
  ```metal
  // beam search GPU 커널
  kernel void cagra_beam_search(
    device const float* dataset,
    device const float* queries,
    device uint32_t* neighbors,
    device float* distances,
    constant CagraSearchParams& params,
    uint tid [[thread_position_in_grid]]
  ) { ... }
  ```
  - Red: 커널 파일 없으면 빌드 실패 (CMakeLists.txt 강제)
  - Green: 커널 컴파일 성공

- [ ] `src/metal_search.mm` 작성 (Objective-C++)
  ```objc
  #import <Metal/Metal.h>
  // MTLDevice → MTLCommandQueue → MTLComputePipelineState
  // MTLBuffer (unified memory, no copy needed)
  // MTLComputeCommandEncoder dispatch
  ```
  - Red: dispatch 없으면 테스트 실패
  - Green: GPU dispatch 확인 (mock command encoder 또는 실측 타이밍)

- [ ] `cuvs::neighbors::cagra::search()` → `metal_search.mm` 연결
  - `src/cagra.cpp`의 stub을 Metal 실제 구현으로 교체

### FAISS 연결 (AC 1, 2)

- [ ] `CMakeLists.txt`에 `Metal.framework`, `CoreFoundation.framework` 링크 추가
- [ ] FAISS `FAISS_CUVS_LIBRARY` → `libcuvs_silicon` 경로 연결
- [ ] `faiss/gpu/test/TestGpuIndexCagra` 컴파일 + 실행

---

## 우선순위 3 — 벤치마크 (AC 4, 5, 6, 13)

SIFT-1M 다운로드 + Metal 검색 완료 후 진행.

- [ ] `cuvs_silicon/benchmark.py` — SIFT-1M 실제 실행 연결
  - 5 warmup + 10 measured, batch_size=100
  - p50/p99 latency, QPS 계산
- [ ] CPU HNSW baseline (hnswlib): M=16, efConstruction=200, efSearch=128, single-threaded
- [ ] 동일 하드웨어에서 Metal GPU QPS > CPU HNSW QPS 확인

---

## 우선순위 4 — 비교 검증 (AC 11, 12)

- [ ] Faiss-mlx 설치: `pip install faiss-mlx` (버전 기록)
- [ ] IVFFlat nprobe-tuned 또는 Flat으로 recall@10 >= 0.90 달성 설정
- [ ] cuvs-silicon vs Faiss-mlx: recall, QPS, p99 비교 리포트

---

## 완료 체크

```
[ ] AC 3  SIFT-1M 로더
[ ] AC 8  하드웨어 보고
[ ] AC 9  macOS/Xcode 버전
[ ] AC 10 FAISS v1.8.0 호환
[ ] AC 1  FAISS 컴파일
[ ] AC 2  TestGpuIndexCagra 통과
[ ] AC 7  Metal GPU QPS > CPU HNSW
[ ] AC 4  recall@10 >= 0.90
[ ] AC 5  벤치마크 프로토콜
[ ] AC 6  CPU HNSW baseline
[ ] AC 13 메트릭 보고
[ ] AC 11 Faiss-mlx 벤치마크
[ ] AC 12 Faiss-mlx 능가
```

13개 모두 체크 → `ooo evaluate` → APPROVED → Done.

---

## 주요 설계 결정 기록

| 결정 | 이유 |
|---|---|
| cuVS API 호환성을 생태계 진입 전략으로 | FAISS/Milvus/Lucene이 모두 cuVS 인터페이스 사용 → cuvs-silicon가 cuVS API 구현 시 전체 생태계의 Apple Silicon 백엔드가 됨 |
| FAISS를 1차 통합 타겟으로 | C++ 직접 통합, macOS 로컬 빌드 용이, 벤치마크 내장 |
| Faiss-mlx를 성능 비교 대상으로 | Apple Silicon 최고 수준 기존 GPU ANN — 이를 능가해야 차별성 증명 |
| MPS는 거리 계산에만 허용 | beam search 그래프 순회는 불규칙 메모리 접근으로 MLX/MPS 부적합 |
| CPU-assisted 인덱스 빌드 허용 (MVP) | Metal Compute Shader 빌드(nn-descent)는 v1.0 목표 |
| CPU fallback 검색 완전 금지 | `std::partial_sort` 등 CPU 루프는 AC 위반 |
