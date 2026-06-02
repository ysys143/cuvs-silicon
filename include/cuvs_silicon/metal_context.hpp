#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace cuvs_silicon {

// Process-level Metal GPU context.
// All Objective-C types are hidden behind the PIMPL Impl struct
// (defined in metal_context.mm) so this header is safe to include from
// plain C++ translation units without triggering ObjC compilation.
class MetalContext {
public:
    // Meyers singleton — thread-safe first-call initialization (C++11).
    static MetalContext& instance();

    // Returns false if MTLCreateSystemDefaultDevice() returned nil
    // (e.g., running on non-Apple or in a headless CI environment).
    bool is_available() const noexcept;

    // Apple Silicon chip model string, e.g. "Apple M3 Max".
    // Returns empty string if not available.
    std::string chip_model() const;

    // Brute-force GPU search: compute L2 distances for all Q queries
    // against all N base vectors, then select top-K per query on CPU.
    //
    // dataset:       [N x D] float32, row-major
    // queries:       [Q x D] float32, row-major
    // out_neighbors: [Q x K] uint32 output (indices into dataset)
    // out_distances: [Q x K] float32 output (squared L2 distances)
    //
    // Returns the number of Metal GPU kernel dispatches made (== Q).
    // Throws std::runtime_error if the device is unavailable or if
    // a Metal command buffer error occurs.
    uint64_t search_brute_force(
        const float* dataset,       int64_t N, int64_t D,
        const float* queries,       int64_t Q,
        uint32_t*    out_neighbors,
        float*       out_distances,
        int64_t      K);

    // GPU-accelerated KNN graph construction.
    // Uses random-bucket MPS matmul seeding + nn-descent Metal kernel refinement.
    // No N^2 CPU limit — scales to millions of vectors.
    // Returns flat N×G uint32 adjacency list (row-major).
    std::vector<uint32_t> build_knn_graph(
        const float* dataset, int64_t N, int64_t D,
        uint32_t     graph_degree    = 32,
        uint32_t     n_descent_iters = 10,
        int64_t      bucket_size     = 2048);

    // CAGRA graph beam search — requires knn_graph built with build().
    // beam_size: number of candidates in the beam (default 64)
    // max_iterations: max expansion steps (default 100)
    // entry_nodes[q] = the graph node to start beam search from for query q.
    // Pass nullptr to fall back to the random hash heuristic.
    uint64_t search_cagra(
        const float*    dataset,       int64_t N, int64_t D,
        const uint32_t* knn_graph,     uint32_t graph_degree,
        const float*    queries,       int64_t Q,
        uint32_t*       out_neighbors,
        float*          out_distances,
        int64_t         K,
        uint32_t        beam_size    = 128,
        uint32_t        max_iter     = 200,
        const uint32_t* entry_nodes  = nullptr);

    // Run the copy_kernel on src[0..n-1] → dst[0..n-1].
    // Used only by test_metal_copy_kernel to verify pipeline dispatch.
    void run_copy_kernel(const float* src, float* dst, int64_t n);

    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

private:
    MetalContext();
    ~MetalContext();
    struct Impl;
    Impl* impl_;
};

} // namespace cuvs_silicon
