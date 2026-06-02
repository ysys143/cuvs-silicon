// TDD Red: Metal GPU dispatch assertion
//
// 이 테스트는 cagra::search()가 실제로 Metal GPU 커널을 dispatch했는지 검증한다.
// CPU brute-force(std::partial_sort)만으로는 통과할 수 없는 조건이다.
//
// Red   — metal_dispatch_count() == 0 (현재 상태, CPU stub만 있음)
// Green — metal_search.mm의 MTLComputeCommandEncoder dispatch가 카운터를 올림

#include <cassert>
#include <cstdint>
#include <vector>

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/resources.hpp>

static void test_search_dispatches_metal_gpu() {
    constexpr int N = 64;    // base vectors
    constexpr int Q = 4;     // query vectors
    constexpr int D = 8;     // dimensions
    constexpr int K = 4;     // top-k

    std::vector<float> base(N * D, 0.0f);
    for (int i = 0; i < N; ++i)
        base[i * D + (i % D)] = static_cast<float>(i);

    std::vector<float> queries(Q * D, 0.0f);
    for (int q = 0; q < Q; ++q)
        queries[q * D + (q % D)] = 1.0f;

    std::vector<std::uint32_t> neighbors(Q * K, 0);
    std::vector<float> distances(Q * K, 0.0f);

    raft::resources res;
    cuvs::neighbors::cagra::index_params ip;
    cuvs::neighbors::cagra::search_params sp;

    auto idx = cuvs::neighbors::cagra::build(
        res, ip,
        raft::device_matrix_view<const float, std::int64_t>(
            base.data(), N, D));

    // Reset counter before search so prior test runs don't interfere.
    cuvs::neighbors::cagra::reset_metal_dispatch_count();

    cuvs::neighbors::cagra::search(
        res, sp, idx,
        raft::device_matrix_view<const float, std::int64_t>(queries.data(), Q, D),
        raft::device_matrix_view<std::uint32_t, std::int64_t>(neighbors.data(), Q, K),
        raft::device_matrix_view<float, std::int64_t>(distances.data(), Q, K));

    // RED: fails because search() currently uses std::partial_sort (CPU),
    // so metal_dispatch_count() is still 0.
    // GREEN: metal_search.mm increments the counter on MTLComputeCommandEncoder dispatch.
    const auto dispatches = cuvs::neighbors::cagra::metal_dispatch_count();
    assert(dispatches > 0 &&
           "FAIL: cagra::search() did not dispatch a Metal GPU kernel. "
           "CPU fallback is not acceptable per seed constraint.");
}

int main() {
    test_search_dispatches_metal_gpu();
    return 0;
}
