# cuvs-silicon

GPU-accelerated graph-based ANN index builder for Apple Silicon, using Metal.

CAGRA (Critically Accelerated Graph-based Approximate nearest neighbor)는 NVIDIA cuVS의 핵심 알고리즘이다. 이 프로젝트는 동일한 알고리즘을 Apple Silicon Metal로 구현한다.

---

## 왜 이 프로젝트인가

### 현존하는 공백

- Apple Silicon용 GPU-accelerated ANN index builder: 존재하지 않음
- Metal/MLX 기반 graph ANN (HNSW/CAGRA 수준): 존재하지 않음
- CPU HNSW는 있으나 GPU 가속 빌드 없음

### Apple Silicon의 구조적 강점

NVIDIA 대비 다른 특성:

```
NVIDIA A100:
  CPU RAM → [PCIe 64GB/s] → GPU HBM 80GB @ 2TB/s
  대용량 빌드 시 전송이 병목

Apple M4 Ultra:
  CPU = GPU = 192GB 통합 메모리 @ ~800GB/s
  전송 오버헤드 없음. 벡터 데이터가 이미 "GPU 메모리"에 있음
```

192GB 통합 메모리는 80GB HBM 한계인 A100보다 수용 데이터셋이 크다.
연산 처리량은 NVIDIA가 우위지만, 메모리 용량과 전송 비용에서 Apple Silicon이 유리하다.

### 도구 실증 기회

세 가지 AI 지원 커널 개발 도구를 실제 프로젝트에 적용하는 사례 연구:

- **CudaForge** (OptimAI-Lab): Coder + Judge + 하드웨어 피드백 루프 → Metal Instruments 적응
- **OpenEvolve** (algorithmicsuperintelligence): AlphaEvolve 오픈소스 구현, MAP-Elites 기반 커널 구조 탐색
- **autoresearch** (karpathy): 실험 파라미터 자동 최적화

AlphaEvolve가 FlashAttention 커널에서 32.5% speedup을 달성한 것과 동일한 접근을 CAGRA MSL 커널에 적용한다.

---

## CAGRA 알고리즘 개요

### 인덱스 빌드 (핵심 목표)

```
1. Seed graph 생성
   - IVF-PQ로 대략적 k-NN 계산
   - 대규모 병렬 거리 계산 → GPU matmul로 가속

2. Graph refinement
   - 각 노드의 이웃을 beam search로 정제
   - 노드마다 독립적 → GPU threadblock 병렬화
```

CPU HNSW 빌드: 1M 벡터 기준 수십 분
CAGRA GPU 빌드: 수십 초 (NVIDIA 기준)

### 인덱스 빌드가 왜 중요한가

빌드가 빠르면 인덱스 재빌드를 실시간 워크로드에 편입할 수 있다.
스트리밍 데이터, 잦은 업데이트 시나리오에서 결정적 차이.

---

## 피드백 신호 설계

cuVS CAGRA가 reference implementation으로 존재하므로 신호가 명확하다.

```python
def evaluate(msl_kernel_code: str) -> float:
    # 1. 컴파일 (이진)
    if not metal_compile(msl_kernel_code):
        return -inf

    # 2. correctness (이진)
    result = cuvs_silicon_build(test_vectors, kernel=msl_kernel_code)
    if recall_vs_cuvs_reference(result) < 0.99:
        return -inf

    # 3. performance (연속)
    return -build_time_ms(test_vectors)
```

correctness: cuVS CAGRA 결과와 비교 (이진)
performance: 빌드 시간 최소화 (연속)

---

## AI 도구 적용 계획

### 단계별 도구 투입

```
Phase 1 — CudaForge (Metal 적응판)
  목표: 동작하는 MSL 커널 확보
  도구: Coder + Judge + Metal GPU Capture 피드백
  NCU(Nsight) 대신 metal_profiler / Instruments 사용

Phase 2 — OpenEvolve
  목표: 구조적으로 다른 커널 variants 탐색
  도구: MAP-Elites, CudaForge를 mutator로 내장
  탐색 공간: threadgroup 레이아웃, 메모리 접근 패턴, 탐색 전략

Phase 3 — autoresearch
  목표: 실험 파라미터 최적화 및 평가 지표 검증
  도구: 단일 파일 수정 + keep/revert 루프
  대상: M, ef_construction, 데이터셋 크기, 평가 지표 구성
```

### 도구 조합 구조

```python
# Phase 2: CudaForge-as-mutator in OpenEvolve
def evolve_mutate(parent_kernel: str) -> str:
    candidate = parent_kernel
    for _ in range(3):
        profile = metal_profile(candidate)
        if is_good_enough(profile):
            break
        candidate = judge_and_fix(candidate, profile)
    return candidate

openevolve.run(
    initial_kernel=reference_msl,
    mutate_fn=evolve_mutate,
    evaluate_fn=lambda k: -build_time_ms(k),
    correctness_fn=lambda k: recall_vs_cuvs(k) > 0.99
)
```

---

## 로드맵

### Phase 0 — 기반 구축 (1-2주)

- [ ] MLX 설치 및 환경 구성
- [ ] cuVS CAGRA reference 실행 환경 (NVIDIA, GCP 또는 로컬)
- [ ] 평가 데이터셋 준비 (ANN-benchmarks 표준 데이터셋)
- [ ] 기준선 측정: CPU HNSW (hnswlib), cuVS CAGRA (NVIDIA)
- [ ] Metal Instruments 프로파일링 파이프라인

### Phase 1 — 동작하는 MSL 커널 (4-8주)

- [ ] MLX matmul 기반 brute force KNN (Python, 정확성 검증)
- [ ] IVF-PQ seeding을 MLX로 구현
- [ ] k-NN graph 초기 구성 (MLX 조합)
- [ ] Graph refinement beam search — MSL 커널 첫 버전
- [ ] CudaForge 패턴 적용: Coder + Metal Instruments Judge
- [ ] correctness 검증: cuVS reference와 결과 비교

### Phase 2 — 성능 최적화 (2-4개월)

- [ ] OpenEvolve 적용: 커널 구조 탐색
- [ ] Metal threadgroup 최적화 (bank conflict, occupancy)
- [ ] 메모리 접근 패턴 최적화 (unified memory 특성 활용)
- [ ] 빌드 시간 측정: CPU HNSW 대비 speedup 달성

### Phase 3 — 검증 및 문서화

- [ ] autoresearch로 실험 파라미터 최적화
- [ ] ANN-benchmarks 표준 결과 기록
- [ ] "AI 도구로 GPU 커널 개발" 사례 문서화
- [ ] C API 설계 (pg_cuvs MPS 백엔드 연결 준비)

---

## 기술 결정

### Metal vs MLX 역할 분담

| 레이어 | 도구 | 이유 |
|--------|------|------|
| 거리 계산, matmul | MLX | 이미 최적화된 Metal 커널 제공 |
| Graph refinement beam search | MSL 직접 | MLX가 조건부 그래프 탐색 미지원 |
| 인터페이스 | Python | 초기 프로토타입, 이후 C API |

### correctness 기준

cuVS CAGRA와 동일 데이터셋에서 동일 이웃을 찾으면 correct.
recall@10 >= 0.99 (cuVS reference 대비).

### 수용하는 트레이드오프

- 빌드 속도: NVIDIA CAGRA보다 느릴 수 있음 (연산 처리량 차이)
- 검색 QPS: 초기에는 목표 아님, 빌드 정확성이 우선
- 지원 차원: 초기 128, 256, 1536차원 고정

---

## 참고

- [CAGRA 논문](https://arxiv.org/pdf/2308.15136)
- [NVIDIA cuVS](https://github.com/rapidsai/cuvs)
- [Apple MLX](https://github.com/ml-explore/mlx)
- [CudaForge](https://github.com/OptimAI-Lab/CudaForge)
- [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve)
- [autoresearch](https://github.com/karpathy/autoresearch)
- [ANN Benchmarks](https://ann-benchmarks.com)
- [AlphaEvolve 논문](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)
