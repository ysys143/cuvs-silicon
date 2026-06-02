// Cohere Wikipedia embedding validation for Metal CAGRA brute-force search.
//
// Loads base/query/ground-truth binary files produced by
// scripts/download_cohere_wiki.py, runs Metal GPU search and CPU HNSW baseline,
// then reports recall@k, QPS, and p99 latency for both.
//
// Usage:
//   ./test_cohere_wiki_validation \
//       data/cohere_wiki_base.bin \
//       data/cohere_wiki_queries.bin \
//       data/cohere_wiki_gt.bin
//
// Binary format (all files):
//   int32 rows, int32 cols
//   rows * cols * sizeof(element) bytes  (float32 for base/queries, int32 for gt)

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/resources.hpp>
#include <cuvs_silicon/metal_context.hpp>

// ── Binary file loader ────────────────────────────────────────────────────

template<typename T>
static std::vector<T> load_bin(const std::string& path,
                                int32_t& rows, int32_t& cols) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    f.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&cols), sizeof(int32_t));
    std::vector<T> data(static_cast<size_t>(rows) * cols);
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(data.size() * sizeof(T)));
    if (!f) throw std::runtime_error("Truncated file: " + path);
    return data;
}

// ── Recall@k computation ─────────────────────────────────────────────────

static double compute_recall(const std::vector<uint32_t>& result,
                              const std::vector<int32_t>&  gt,
                              int64_t Q, int64_t K) {
    int64_t hits = 0;
    for (int64_t q = 0; q < Q; ++q) {
        for (int64_t r = 0; r < K; ++r) {
            const uint32_t predicted = result[static_cast<size_t>(q * K + r)];
            // Check if predicted index appears anywhere in ground truth row
            for (int64_t g = 0; g < K; ++g) {
                if (static_cast<int32_t>(predicted) ==
                        gt[static_cast<size_t>(q * K + g)]) {
                    ++hits;
                    break;
                }
            }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(Q * K);
}

// ── Timing helper ─────────────────────────────────────────────────────────

using Clock = std::chrono::high_resolution_clock;

struct BenchResult {
    double recall_at_k;
    double qps;
    double p50_ms;
    double p99_ms;
};

static BenchResult run_metal_bench(
        const float*  base,    int64_t N, int64_t D,
        const float*  queries, int64_t Q,
        const int32_t* gt,
        int64_t K,
        int warmup_iters, int measure_iters) {

    // Build CAGRA index (GPU graph construction)
    raft::resources res;
    cuvs::neighbors::cagra::index_params ip;
    ip.graph_degree = 64;

    std::printf("  Building CAGRA graph (GPU)...\n"); fflush(stdout);
    std::atomic<bool> build_done{false};
    std::thread watchdog([&build_done]() {
        for (int i = 0; i < 120; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (build_done.load()) return;
        }
        std::fprintf(stderr, "\n[TIMEOUT] Build exceeded 120s. Aborting.\n");
        _exit(1);
    });
    const auto t_build0 = Clock::now();
    auto idx = cuvs::neighbors::cagra::build(
        res, ip,
        raft::device_matrix_view<const float, std::int64_t>(base, N, D));
    build_done.store(true);
    watchdog.join();
    const double build_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t_build0).count();
    std::printf("  Build time: %.2fs  graph_degree=%u  has_graph=%s\n",
                build_ms / 1000.0,
                idx.graph_degree(),
                idx.graph_degree() > 0 ? "yes" : "no");

    cuvs::neighbors::cagra::search_params sp;
    sp.itopk_size     = 512;
    sp.max_iterations = 600;

    std::vector<uint32_t> neighbors(static_cast<size_t>(Q * K));
    std::vector<float>    distances(static_cast<size_t>(Q * K));

    auto do_search = [&]() {
        cuvs::neighbors::cagra::search(
            res, sp, idx,
            raft::device_matrix_view<const float, std::int64_t>(queries, Q, D),
            raft::device_matrix_view<uint32_t, std::int64_t>(neighbors.data(), Q, K),
            raft::device_matrix_view<float,    std::int64_t>(distances.data(), Q, K));
    };

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) do_search();

    cuvs::neighbors::cagra::reset_metal_dispatch_count();

    // Measure
    std::vector<double> latencies_ms;
    latencies_ms.reserve(static_cast<size_t>(measure_iters));
    for (int i = 0; i < measure_iters; ++i) {
        const auto t0 = Clock::now();
        do_search();
        const auto t1 = Clock::now();
        latencies_ms.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    const uint64_t dispatches = cuvs::neighbors::cagra::metal_dispatch_count();
    std::printf("  Metal dispatch count: %llu\n",
                static_cast<unsigned long long>(dispatches));

    const double recall = compute_recall(neighbors,
                                          std::vector<int32_t>(gt, gt + Q * K),
                                          Q, K);

    std::sort(latencies_ms.begin(), latencies_ms.end());
    const double total_ms = std::accumulate(latencies_ms.begin(),
                                             latencies_ms.end(), 0.0);
    const double p50 = latencies_ms[static_cast<size_t>(measure_iters * 0.50)];
    const double p99 = latencies_ms[static_cast<size_t>(measure_iters * 0.99)];
    const double qps = static_cast<double>(Q * measure_iters) /
                       (total_ms / 1000.0);

    return {recall, qps, p50, p99};
}

// ── CPU brute-force (reference) ───────────────────────────────────────────

static BenchResult run_cpu_bench(
        const float*  base,    int64_t N, int64_t D,
        const float*  queries, int64_t Q,
        const int32_t* gt,
        int64_t K,
        int warmup_iters, int measure_iters) {

    std::vector<uint32_t> neighbors(static_cast<size_t>(Q * K));
    std::vector<float>    distances(static_cast<size_t>(Q * K));

    auto do_search = [&]() {
        std::vector<std::pair<float, uint32_t>> cands(static_cast<size_t>(N));
        for (int64_t q = 0; q < Q; ++q) {
            const float* qv = queries + q * D;
            for (int64_t n = 0; n < N; ++n) {
                const float* bv = base + n * D;
                float dist = 0.f;
                for (int64_t d = 0; d < D; ++d) {
                    float delta = qv[d] - bv[d];
                    dist += delta * delta;
                }
                cands[static_cast<size_t>(n)] = {dist, static_cast<uint32_t>(n)};
            }
            const auto end_it = (K < N) ? cands.begin() + K : cands.end();
            std::partial_sort(cands.begin(), end_it, cands.end(),
                [](const auto& a, const auto& b){ return a.first < b.first; });
            for (int64_t k = 0; k < K; ++k) {
                neighbors[static_cast<size_t>(q * K + k)] = cands[static_cast<size_t>(k)].second;
                distances[static_cast<size_t>(q * K + k)] = cands[static_cast<size_t>(k)].first;
            }
        }
    };

    for (int i = 0; i < warmup_iters; ++i) do_search();

    std::vector<double> latencies_ms;
    latencies_ms.reserve(static_cast<size_t>(measure_iters));
    for (int i = 0; i < measure_iters; ++i) {
        const auto t0 = Clock::now();
        do_search();
        const auto t1 = Clock::now();
        latencies_ms.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    const double recall = compute_recall(neighbors,
                                          std::vector<int32_t>(gt, gt + Q * K),
                                          Q, K);

    std::sort(latencies_ms.begin(), latencies_ms.end());
    const double total_ms = std::accumulate(latencies_ms.begin(),
                                             latencies_ms.end(), 0.0);
    const double p50 = latencies_ms[static_cast<size_t>(measure_iters * 0.50)];
    const double p99 = latencies_ms[static_cast<size_t>(measure_iters * 0.99)];
    const double qps = static_cast<double>(Q * measure_iters) /
                       (total_ms / 1000.0);

    return {recall, qps, p50, p99};
}

// ── Main ──────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::fprintf(stderr,
            "Usage: %s <base.bin> <queries.bin> <gt.bin> [K=10] [warmup=3] [measure=10]\n",
            argv[0]);
        return 1;
    }

    const std::string base_path    = argv[1];
    const std::string queries_path = argv[2];
    const std::string gt_path      = argv[3];
    const int64_t K           = argc > 4 ? std::stoll(argv[4]) : 10;
    const int     warmup_iter = argc > 5 ? std::stoi(argv[5]) : 3;
    const int     measure_iter = argc > 6 ? std::stoi(argv[6]) : 10;

    int32_t base_rows, base_cols, q_rows, q_cols, gt_rows, gt_cols;
    const auto base    = load_bin<float>  (base_path,    base_rows, base_cols);
    const auto queries = load_bin<float>  (queries_path, q_rows,    q_cols);
    const auto gt      = load_bin<int32_t>(gt_path,      gt_rows,   gt_cols);

    assert(base_cols == q_cols && "dimension mismatch");
    assert(gt_rows == q_rows   && "ground truth row count mismatch");

    const int64_t N = base_rows, D = base_cols, Q = q_rows;
    std::printf("\nCohere Wikipedia Validation\n");
    std::printf("  Base vectors:  %lld x %lld (%.1f MB)\n",
                (long long)N, (long long)D,
                static_cast<double>(N * D * 4) / 1e6);
    std::printf("  Query vectors: %lld x %lld\n", (long long)Q, (long long)D);
    std::printf("  K = %lld  warmup=%d  measure=%d\n\n",
                (long long)K, warmup_iter, measure_iter);

    // ── Metal GPU search ──
    std::printf("[Metal GPU brute-force]\n");
    BenchResult metal_res;
    try {
        metal_res = run_metal_bench(base.data(), N, D,
                                     queries.data(), Q,
                                     gt.data(), K,
                                     warmup_iter, measure_iter);
        std::printf("  recall@%lld = %.4f\n",  (long long)K, metal_res.recall_at_k);
        std::printf("  QPS        = %.1f\n",  metal_res.qps);
        std::printf("  p50 ms     = %.2f\n",  metal_res.p50_ms);
        std::printf("  p99 ms     = %.2f\n",  metal_res.p99_ms);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "  Metal search failed: %s\n", e.what());
        return 1;
    }

    // ── CPU brute-force (skip for large N — O(Q×N×D) takes too long) ──
    std::printf("\n[CPU brute-force]\n");
    constexpr int64_t CPU_BENCH_LIMIT = 50000;
    if (N > CPU_BENCH_LIMIT) {
        std::printf("  skipped (N=%lld > %lld)\n", (long long)N, (long long)CPU_BENCH_LIMIT);
    } else {
        const auto cpu_res = run_cpu_bench(base.data(), N, D,
                                            queries.data(), Q,
                                            gt.data(), K,
                                            warmup_iter, measure_iter);
        std::printf("  recall@%lld = %.4f\n", (long long)K, cpu_res.recall_at_k);
        std::printf("  QPS        = %.1f\n", cpu_res.qps);
        std::printf("  p50 ms     = %.2f\n", cpu_res.p50_ms);
        std::printf("  p99 ms     = %.2f\n", cpu_res.p99_ms);

        std::printf("\n[Comparison]\n");
        std::printf("  Metal QPS / CPU QPS = %.2fx\n",
                    metal_res.qps / cpu_res.qps);
        std::printf("  Metal p99 / CPU p99 = %.2fx\n",
                    metal_res.p99_ms / cpu_res.p99_ms);
        const bool metal_faster = metal_res.qps > cpu_res.qps;
        std::printf("  Metal GPU %s CPU brute-force\n",
                    metal_faster ? "FASTER than" : "SLOWER than");
    }

    assert(metal_res.recall_at_k >= 0.99 &&
           "FAIL: recall@k must be >= 0.99");
    std::printf("\nPASS: recall@%lld = %.4f >= 0.99\n",
                (long long)K, metal_res.recall_at_k);

    return 0;
}
