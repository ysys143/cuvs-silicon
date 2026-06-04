# Multi-CTA + IVF_PQ 구현 교훈

**작성일**: 2026-06-05  
**세션 범위**: Multi-CTA beam search + IVF_PQ 빌드 구현 시도  
**최종 상태**: 100K PASS, 1M 빌드 완료(2714s), recall 0.72

---

## 최종 결과

| N | 이전 | 이번 세션 후 |
|---|---|---|
| 100K build | 75s, recall=0.9927 | 149s, recall=0.9960 PASS |
| 1M build | 타임아웃(∞) | 2714s 완료, recall=0.72 |

---

## 교훈 1: Multi-CTA는 memory-bound workload에서 역효과

**시도**: Q=1 CAGRA 레이턴시 개선을 위해 N_TEAMS=16 개의 threadgroup을 병렬 dispatch.

**결과**: Q=1 레이턴시 54ms → **92ms로 악화**.

**원인**: CAGRA beam search는 compute-bound가 아니라 **memory-bound** workload다. 그래프(25MB)와 데이터셋(400MB)에 대한 랜덤 메모리 접근이 병목. N_TEAMS=16이면 16개 팀이 동시에 같은 캐시 라인을 놓고 경쟁 → cache thrashing.

**교훈**: Multi-CTA는 compute-bound 커널에서만 유효하다. CUDA CUVS가 빠른 이유는 multi-CTA가 아니라 **HBM2e 메모리 대역폭(2TB/s)**이다. M3 Max의 400GB/s로는 단일 쿼리 레이턴시를 근본적으로 개선할 수 없다.

**코드**: `cagra_beam_search_multi_cta` 커널은 구현됐지만 활성화 안 함 (N_TEAMS=1 고정).

---

## 교훈 2: IVF_PQ 병목은 알고리즘이 아니라 하드웨어 대역폭

**시도**: O(N^1.5) per-cluster sgemm을 O(N log N) PQ LUT lookup으로 교체.

**실측 타임스탬프 (N=1M)**:
```
K-means:       341s  ← chunk=4096 시 326s였는데 비슷
PQ training:   764s  ← 10 iters × 8 subspaces × N=1M
CPU seeding:  1548s  ← PQ lookup × 229B pairs
Total:        2714s  
```

**PQ 세딩이 1548s인 이유**: 1000 clusters × 4064 own × 7064 probe × 8 LUT reads = 229B reads. LUT(2MB)는 L2에 들어가지만 pq_codes(8MB)는 L3에서 partial miss. 이론값 229s 대비 실측 1548s = **6.7× overhead** (cache pressure, branch misprediction, 함수 호출 오버헤드 누적).

CUDA CUVS는 같은 작업을 22s에 완료: HBM2e(2TB/s)가 CPU DRAM(50GB/s) 대비 40× 더 빠르기 때문.

**교훈**: IVF_PQ는 알고리즘적으로 맞지만, M3 Max CPU/AMX의 메모리 대역폭으로는 CUDA 수준의 속도를 달성할 수 없다. GPU-native 구현(Metal 커널에서 PQ lookup)이 필요하다.

---

## 교훈 3: GPU 세딩 커널 버그 — PQ distance scale 불일치

**시도**: `pq_cluster_seeding` Metal 커널로 GPU에서 직접 PQ 거리 계산 + graph 기록.

**결과**: recall = **0.0038** (랜덤보다 나쁨).

**원인 (두 단계)**:

1단계 버그: `graph_dist`에 PQ 근사 거리(~0.01-0.1)가 저장된 상태로 nn-descent 실행.  
nn-descent는 `exact_L2(2-hop) < worst_pq_dist`를 비교하는데, exact L2(~1.0)이 항상 PQ dist(~0.05)보다 크게 보여서 **nn-descent가 아무것도 개선 못 함**.

2단계 버그: graph_dist를 INFINITY로 초기화하면 nn-descent race condition 발생.  
```metal
// 여러 thread가 동시에 worst_g = G-1 (같은 INFINITY 슬롯)을 찾아 덮어씀
float worst = 0.f; // worst_g = G-1 (INFINITY)
if (dist < worst) { graph[row + G-1] = nbr; } // 모든 thread가 G-1에 씀
```
결과: 모든 노드의 graph slot G-1만 업데이트되고 나머지는 UINT_MAX 유지 → 완전히 망가진 그래프.

**해결**: CPU PQ 세딩 + GPU L2 재계산 커널(`recompute_graph_distances`)로 교체.  
- CPU PQ: 정확한 topology를 찾음 (올바른 이웃 인덱스)
- GPU L2 재계산: 정확한 L2 거리로 graph_dist를 수정
- 그 후 nn-descent: 올바른 스케일로 비교 가능

**교훈**: Graph 알고리즘에서 **거리 스케일 일관성**이 핵심이다. PQ 근사 거리와 exact L2 거리를 같은 graph_dist 배열에 혼용하면 비교 로직이 깨진다. nn-descent의 암묵적 가정: "graph_dist는 항상 동일한 거리 함수에서 나온 값"이다.

---

## 교훈 4: N 규모별 최적 전략이 다르다

**최종 결정**:
- N ≤ 200K: **exact sgemm 세딩** (AMX 캐시 내 처리, recall ~0.99)
- N > 200K: **CPU PQ 세딩 + GPU L2 재계산** (대규모 필수, recall ~0.72)

exact sgemm이 100K에서 빠른 이유: chunk=2048일 때 A(8MB)+C(8MB)+B(4MB) = 20MB < L3(32MB) → cache hit.

PQ가 필요한 규모: N=1M에서는 per-cluster sgemm 자체가 16MB/chunk로 L3 초과 → DRAM bottleneck → 900s+ timeout.

**chunk 크기의 영향**:
- chunk=4096: K-means E-step A(16MB)+C(16MB)=32MB ≥ L3 → DRAM bottleneck → K-means 326s
- chunk=2048: A(8MB)+C(8MB)+B(4MB)=20MB < L3 → K-means 245-341s (L3 hit)

---

## 교훈 5: GPU recompute_graph_distances 커널이 유용하다

`recompute_graph_distances` Metal 커널:
- N threadgroups × G threads
- 각 thread: 하나의 그래프 엣지에 대해 정확한 L2 재계산
- 100K: ~1s (vs CPU 재계산 48s)

이 패턴은 PQ 세딩뿐 아니라 다양한 근사 초기화 이후 거리 정규화에 재사용 가능하다.

---

## 다음 단계 (만약 계속한다면)

**1M recall 0.72 → 0.99 로 올리려면:**
- Option A: `recompute_graph_distances` 이후 1-2 nn-descent GPU 이터레이션 추가
  - 문제: 1M nn-descent 1회 = ~900s (DRAM random access bottleneck)
  - 총 빌드: 2714 + 900 = 3614s > 3600s 타임아웃
  - 해결: 타임아웃 추가 증가 OR nn-descent 커널 최적화 필요
  
- Option B: PQ_M=16으로 세딩 품질 향상
  - PQ training 더 느려짐 (이미 764s)
  - 하지만 nn-descent 없이도 더 좋은 topology 보장
  
- Option C: CPU PQ 세딩을 Metal GPU 커널로 이식
  - GPU PQ lookup: pq_codes(8MB)와 LUT(2MB) 모두 GPU L2/SLC에 상주
  - GPU의 높은 메모리 대역폭으로 1548s → 목표 <100s
  - 이것이 CUDA CUVS가 하는 방식

**K-means M-step 병목 (341s):**
- M-step cblas_saxpy 패턴: N × cblas_saxpy(D, dataset[i], centroid[k]) - scatter-add
- Accelerate AMX는 이 패턴을 효율적으로 처리 못함 (scatter pattern)
- 해결: cluster별로 모아서 처리 (데이터 재정렬) or GPU atomic scatter-add 커널
