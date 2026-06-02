// metal_search.mm — Top-K CPU helper shared by metal_context.mm.
// Compiled as OBJCXX so it lives in the same metal_cagra_context library,
// but contains only standard C++ (no Metal/ObjC calls here).
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

// Selects K nearest neighbors from a flat distance array of length N.
// Writes sorted (ascending) indices and distances to out_* buffers.
// Called on CPU after each GPU kernel dispatch in metal_context.mm.
void top_k_cpu(const float* distances, int64_t N,
               uint32_t* out_neighbors, float* out_distances, int64_t K) {
    std::vector<uint32_t> idx(static_cast<size_t>(N));
    std::iota(idx.begin(), idx.end(), 0u);
    const int64_t valid_k = (K < N) ? K : N;
    std::partial_sort(idx.begin(), idx.begin() + valid_k, idx.end(),
        [&](uint32_t a, uint32_t b) {
            if (distances[a] != distances[b]) return distances[a] < distances[b];
            return a < b;  // tie-break by ascending label index (FAISS contract)
        });
    for (int64_t i = 0; i < K; ++i) {
        if (i < N) {
            out_neighbors[i] = idx[static_cast<size_t>(i)];
            out_distances[i] = distances[idx[static_cast<size_t>(i)]];
        } else {
            // K > N: overflow slots filled per FAISS contract
            out_neighbors[i] = std::numeric_limits<uint32_t>::max();
            out_distances[i] = std::numeric_limits<float>::infinity();
        }
    }
}
