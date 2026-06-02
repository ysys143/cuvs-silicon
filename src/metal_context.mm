#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuvs_silicon/metal_context.hpp>

void top_k_cpu(const float* distances, int64_t N,
               uint32_t* out_neighbors, float* out_distances, int64_t K);

namespace cuvs_silicon {

// ── PIMPL ─────────────────────────────────────────────────────────────────

struct MetalContext::Impl {
    id<MTLDevice>               device   = nil;
    id<MTLCommandQueue>         queue    = nil;
    id<MTLLibrary>              library  = nil;
    id<MTLComputePipelineState> pso_copy         = nil;
    id<MTLComputePipelineState> pso_beam_search  = nil;
    id<MTLComputePipelineState> pso_nn_descent   = nil;
    bool                        available = false;
    std::string                 chip_model_str;

    // Dataset norm cache — invalidated when dataset pointer or shape changes.
    const float*       cached_dataset_ptr = nullptr;
    int64_t            cached_N = 0, cached_D = 0;
    std::vector<float> cached_dataset_norms;

    // Reusable cross-product buffer — avoids 4MB+ alloc per search call.
    std::vector<float> cross_buf;
    int64_t            cross_buf_Q = 0, cross_buf_N = 0;
};

// ── Constructor ───────────────────────────────────────────────────────────

MetalContext::MetalContext() : impl_(new Impl) {
    @autoreleasepool {
        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) { impl_->available = false; return; }

        impl_->queue = [impl_->device newCommandQueue];
        if (!impl_->queue) { impl_->available = false; return; }

        NSString* path = @CAGRA_METALLIB_PATH;
        NSURL*    url  = [NSURL fileURLWithPath:path];
        NSError*  err  = nil;
        impl_->library = [impl_->device newLibraryWithURL:url error:&err];
        if (!impl_->library) { impl_->available = false; return; }

        auto make_pso = [&](NSString* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [impl_->library newFunctionWithName:name];
            if (!fn) return nil;
            NSError* e = nil;
            return [impl_->device newComputePipelineStateWithFunction:fn error:&e];
        };
        impl_->pso_copy        = make_pso(@"copy_kernel");
        impl_->pso_beam_search = make_pso(@"cagra_beam_search");
        impl_->pso_nn_descent  = make_pso(@"nn_descent");

        impl_->chip_model_str = impl_->device.name.UTF8String;
        impl_->available      = true;
    }
}

MetalContext::~MetalContext() { delete impl_; }

MetalContext& MetalContext::instance() {
    static MetalContext ctx;
    return ctx;
}

bool MetalContext::is_available() const noexcept { return impl_->available; }
std::string MetalContext::chip_model() const { return impl_->chip_model_str; }

// ── Buffer helper ─────────────────────────────────────────────────────────

static id<MTLBuffer> make_shared_buf(id<MTLDevice> dev,
                                      const void* data, NSUInteger bytes) {
    if ((reinterpret_cast<uintptr_t>(data) & 0xFFF) == 0)
        return [dev newBufferWithBytesNoCopy:(void*)data length:bytes
                options:MTLResourceStorageModeShared deallocator:nil];
    return [dev newBufferWithBytes:data length:bytes
            options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> make_empty_buf(id<MTLDevice> dev, NSUInteger bytes) {
    return [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

// ── MPS-based brute-force search ──────────────────────────────────────────
//
// L2(q,d) = ||q-d||^2 = ||q||^2 - 2*(q·d) + ||d||^2
//
// Step 1: cross[Q×N] = queries[Q×D] @ dataset[N×D]^T  (MPS matmul)
// Step 2: dist[q,n]  = q_norms[q] - 2*cross[q,n] + d_norms[n]
// Step 3: top-K per row (CPU)

static std::vector<float> compute_row_norms(const float* mat,
                                             int64_t rows, int64_t cols) {
    std::vector<float> norms(static_cast<size_t>(rows));
    for (int64_t r = 0; r < rows; ++r) {
        float s = 0.f;
        const float* row = mat + r * cols;
        for (int64_t c = 0; c < cols; ++c) s += row[c] * row[c];
        norms[static_cast<size_t>(r)] = s;
    }
    return norms;
}

uint64_t MetalContext::search_brute_force(
        const float* dataset, int64_t N, int64_t D,
        const float* queries,  int64_t Q,
        uint32_t*    out_neighbors,
        float*       out_distances,
        int64_t      K) {

    if (!impl_->available)
        throw std::runtime_error("Metal device not available");

    {

        // ── Accelerate BLAS: cross[Q×N] = queries[Q×D] @ dataset[N×D]^T ──
        // Uses Apple AMX hardware via Accelerate framework — same path as MLX.
        // cblas_sgemm: C = alpha*A*B^T + beta*C
        //   A = queries  (Q×D, lda=D)
        //   B = dataset  (N×D, ldb=D, transposed)
        //   C = cross    (Q×N, ldc=N)
        if (impl_->cross_buf_Q != Q || impl_->cross_buf_N != N)
            impl_->cross_buf.resize(static_cast<size_t>(Q * N));
        impl_->cross_buf_Q = Q;
        impl_->cross_buf_N = N;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)Q, (int)N, (int)D,
                    1.0f, queries, (int)D, dataset, (int)D,
                    0.0f, impl_->cross_buf.data(), (int)N);

        // ── L2 distances: dist[q,n] = q_norm - 2*cross + d_norm ──────────
        const auto q_norms = compute_row_norms(queries, Q, D);

        if (impl_->cached_dataset_ptr != dataset ||
            impl_->cached_N != N || impl_->cached_D != D) {
            impl_->cached_dataset_norms  = compute_row_norms(dataset, N, D);
            impl_->cached_dataset_ptr    = dataset;
            impl_->cached_N              = N;
            impl_->cached_D              = D;
        }
        const auto& d_norms = impl_->cached_dataset_norms;

        const float* cross_ptr = impl_->cross_buf.data();
        std::vector<float> dist_row(static_cast<size_t>(N));

        for (int64_t q = 0; q < Q; ++q) {
            const float qn = q_norms[static_cast<size_t>(q)];
            const float* cross_row = cross_ptr + q * N;
            for (int64_t n = 0; n < N; ++n) {
                dist_row[static_cast<size_t>(n)] =
                    qn - 2.f * cross_row[n] + d_norms[static_cast<size_t>(n)];
            }
            top_k_cpu(dist_row.data(), N,
                      out_neighbors + q * K,
                      out_distances  + q * K,
                      K);
        }
    }
    return 1;  // single batched dispatch
}

// ── copy_kernel dispatch (P3 pipeline verifier) ───────────────────────────

void MetalContext::run_copy_kernel(const float* src, float* dst, int64_t n) {
    if (!impl_->available)
        throw std::runtime_error("Metal device not available");
    if (!impl_->pso_copy)
        throw std::runtime_error("copy_kernel pipeline not available");

    @autoreleasepool {
        auto src_buf = make_shared_buf(impl_->device, src, (NSUInteger)(n * sizeof(float)));
        auto dst_buf = make_empty_buf(impl_->device, (NSUInteger)(n * sizeof(float)));

        auto cmd = [impl_->queue commandBuffer];
        auto enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:impl_->pso_copy];
        [enc setBuffer:src_buf offset:0 atIndex:0];
        [enc setBuffer:dst_buf offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(std::min((int64_t)256, n), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error)
            throw std::runtime_error(cmd.error.localizedDescription.UTF8String);

        std::copy_n((const float*)dst_buf.contents, n, dst);
    }
}

// ── KNN graph build: random-bucket MPS seeding + nn-descent ──────────────

std::vector<uint32_t> MetalContext::build_knn_graph(
        const float* dataset, int64_t N, int64_t D,
        uint32_t G, uint32_t n_descent_iters, int64_t bucket_size_hint) {

    if (!impl_->available)
        throw std::runtime_error("Metal device not available");

    // For small N, use full N×N seeding for exact graph quality.
    // For large N, use multi-pass bucketing (3 passes with different permutations).
    constexpr int64_t FULL_SEEDING_LIMIT = 50000;
    const int64_t bsz = (N <= FULL_SEEDING_LIMIT)
        ? N
        : std::min(bucket_size_hint,
                   std::max((int64_t)G * 32, (int64_t)8192));
    const int n_seed_passes = (N <= FULL_SEEDING_LIMIT) ? 1 : 3;

    // ── Phase 1: Random-bucket seeding ────────────────────────────────
    // Shuffle vector indices, process in buckets of ~bsz.
    // For each bucket compute exact k-NN via MPS matmul.

    std::vector<uint32_t> graph(static_cast<size_t>(N) * G,
                                 std::numeric_limits<uint32_t>::max());
    std::vector<float> graph_dist(static_cast<size_t>(N) * G,
                                   std::numeric_limits<float>::infinity());

    // Row norms (needed for L2 from dot-product)
    std::vector<float> norms(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i) {
        float s = 0.f;
        const float* v = dataset + i * D;
        for (int64_t d = 0; d < D; ++d) s += v[d] * v[d];
        norms[static_cast<size_t>(i)] = s;
    }

    // Multi-pass seeding: each pass uses a different random permutation.
    // Passes > 1 are used for large N to increase graph coverage.
    std::vector<int64_t> perm(static_cast<size_t>(N));
    for (int pass = 0; pass < n_seed_passes; ++pass) {
    std::iota(perm.begin(), perm.end(), 0LL);
    uint64_t rng = 0x123456789ABCDEFULL + static_cast<uint64_t>(pass) * 0xDEADBEEFCAFEULL;
    for (int64_t i = N - 1; i > 0; --i) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        int64_t j = (int64_t)(rng >> 33) % (i + 1);
        std::swap(perm[static_cast<size_t>(i)], perm[static_cast<size_t>(j)]);
    }

    // Process buckets for this pass
    @autoreleasepool {
        for (int64_t start = 0; start < N; start += bsz) {
            const int64_t end = std::min(start + bsz, N);
            const int64_t Bs  = end - start;

            // Gather bucket vectors
            std::vector<float> bucket_data(static_cast<size_t>(Bs * D));
            for (int64_t b = 0; b < Bs; ++b) {
                const int64_t idx = perm[static_cast<size_t>(start + b)];
                std::copy_n(dataset + idx * D,
                            static_cast<size_t>(D),
                            bucket_data.data() + b * D);
            }

            // MPS matmul: cross[Bs×Bs] = bucket[Bs×D] @ bucket^T
            std::vector<float> cross(static_cast<size_t>(Bs * Bs));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        (int)Bs, (int)Bs, (int)D,
                        1.0f, bucket_data.data(), (int)D,
                        bucket_data.data(), (int)D,
                        0.0f, cross.data(), (int)Bs);

            // For each vector in bucket, find top-G neighbors
            std::vector<uint32_t> idx_buf(static_cast<size_t>(Bs));
            for (int64_t b = 0; b < Bs; ++b) {
                const int64_t vi = perm[static_cast<size_t>(start + b)];
                const float ni  = norms[static_cast<size_t>(vi)];
                const float* cr = cross.data() + b * Bs;

                std::iota(idx_buf.begin(), idx_buf.end(), 0u);
                idx_buf[static_cast<size_t>(b)] =
                    idx_buf[static_cast<size_t>(Bs - 1)];
                const int64_t valid = Bs - 1;
                const int64_t take  = std::min((int64_t)G, valid);

                std::partial_sort(idx_buf.begin(),
                                  idx_buf.begin() + take,
                                  idx_buf.begin() + valid,
                    [&](uint32_t a, uint32_t b_idx) {
                        const int64_t ga = perm[static_cast<size_t>(start + a)];
                        const int64_t gb = perm[static_cast<size_t>(start + b_idx)];
                        float da = ni - 2.f * cr[a] + norms[static_cast<size_t>(ga)];
                        float db = ni - 2.f * cr[b_idx] + norms[static_cast<size_t>(gb)];
                        return da < db;
                    });

                const size_t row = static_cast<size_t>(vi) * G;
                for (int64_t k = 0; k < take; ++k) {
                    const int64_t nbr_idx = perm[static_cast<size_t>(
                        start + idx_buf[static_cast<size_t>(k)])];
                    const float nb_ni = norms[static_cast<size_t>(nbr_idx)];
                    const float dist  = ni - 2.f * cr[idx_buf[static_cast<size_t>(k)]]
                                        + nb_ni;
                    // Insert if better than current worst
                    float worst = graph_dist[row];
                    uint32_t worst_g = 0;
                    for (uint32_t g = 1; g < G; ++g) {
                        if (graph_dist[row + g] > worst) {
                            worst   = graph_dist[row + g];
                            worst_g = g;
                        }
                    }
                    if (dist < worst) {
                        graph[row + worst_g]      = static_cast<uint32_t>(nbr_idx);
                        graph_dist[row + worst_g] = dist;
                    }
                }
            }
        }
    }
    } // end pass loop

    // ── Phase 2: nn-descent refinement via Metal kernel ───────────────
    if (impl_->pso_nn_descent && n_descent_iters > 0) {
        @autoreleasepool {
            auto buf_dataset = make_shared_buf(impl_->device, dataset,
                                   (NSUInteger)(N * D * sizeof(float)));
            auto buf_graph = [impl_->device
                newBufferWithBytes:graph.data()
                length:(NSUInteger)(N * G * sizeof(uint32_t))
                options:MTLResourceStorageModeShared];
            auto buf_gdist = [impl_->device
                newBufferWithBytes:graph_dist.data()
                length:(NSUInteger)(N * G * sizeof(float))
                options:MTLResourceStorageModeShared];
            auto buf_improved = make_empty_buf(impl_->device,
                                    (NSUInteger)(N * sizeof(uint32_t)));

            uint32_t uN = (uint32_t)N, uD = (uint32_t)D, uG = G;
            // G threads per node: each thread handles G/G = 1 hop-2 neighbor
            // set. Races on graph[] writes are benign (worst: miss one update).
            const NSUInteger threads = std::min(G, 32u);

            for (uint32_t iter = 0; iter < n_descent_iters; ++iter) {
                memset(buf_improved.contents, 0,
                       static_cast<size_t>(N * sizeof(uint32_t)));

                auto cmd = [impl_->queue commandBuffer];
                auto enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:impl_->pso_nn_descent];
                [enc setBuffer:buf_dataset  offset:0 atIndex:0];
                [enc setBuffer:buf_graph    offset:0 atIndex:1];
                [enc setBuffer:buf_improved offset:0 atIndex:2];
                [enc setBuffer:buf_gdist    offset:0 atIndex:3];
                [enc setBytes:&uN length:4 atIndex:4];
                [enc setBytes:&uD length:4 atIndex:5];
                [enc setBytes:&uG length:4 atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];

                // Check convergence
                const uint32_t* imp =
                    static_cast<const uint32_t*>(buf_improved.contents);
                bool changed = false;
                for (int64_t i = 0; i < N; ++i)
                    if (imp[static_cast<size_t>(i)]) { changed = true; break; }
                if (!changed) break;
            }

            // Read back refined graph
            std::copy_n(static_cast<const uint32_t*>(buf_graph.contents),
                        static_cast<size_t>(N) * G, graph.data());
        }
    }

    return graph;
}

// ── CAGRA beam search dispatch ─────────────────────────────────────────────

uint64_t MetalContext::search_cagra(
        const float*    dataset,    int64_t N, int64_t D,
        const uint32_t* knn_graph,  uint32_t G,
        const float*    queries,    int64_t Q,
        uint32_t*       out_nbrs,
        float*          out_dists,
        int64_t         K,
        uint32_t        beam_size,
        uint32_t        max_iter,
        const uint32_t* entry_nodes) {

    if (!impl_->available)
        throw std::runtime_error("Metal device not available");
    if (!impl_->pso_beam_search)
        throw std::runtime_error("cagra_beam_search kernel not compiled");

    @autoreleasepool {
        const uint32_t visited_words = (static_cast<uint32_t>(N) + 31u) / 32u;

        // ── Persistent buffers (shared across calls) ──────────────────
        auto buf_dataset = make_shared_buf(impl_->device, dataset,
                               (NSUInteger)(N * D * sizeof(float)));
        auto buf_graph   = make_shared_buf(impl_->device, knn_graph,
                               (NSUInteger)((uint64_t)N * G * sizeof(uint32_t)));
        auto buf_queries = make_shared_buf(impl_->device, queries,
                               (NSUInteger)(Q * D * sizeof(float)));
        auto buf_out_nbrs  = make_shared_buf(impl_->device, out_nbrs,
                               (NSUInteger)(Q * K * sizeof(uint32_t)));
        auto buf_out_dists = make_shared_buf(impl_->device, out_dists,
                               (NSUInteger)(Q * K * sizeof(float)));

        // ── Working buffers ───────────────────────────────────────────
        // cand_dists: Q × beam_size floats
        auto buf_cand_dists = make_empty_buf(impl_->device,
                               (NSUInteger)((uint64_t)Q * beam_size * 4u));
        // cand_nodes: Q × beam_size uint32
        auto buf_cand_nodes = make_empty_buf(impl_->device,
                               (NSUInteger)((uint64_t)Q * beam_size * 4u));
        // visited bitfield: Q × visited_words × uint32
        auto buf_visited = make_empty_buf(impl_->device,
                               (NSUInteger)((uint64_t)Q * visited_words * 4u));

        memset(buf_visited.contents, 0,
               (size_t)((uint64_t)Q * visited_words * 4u));

        // Multithreaded: each thread computes distances in parallel,
        // thread 0 merges results into the beam (no atomic races).
        const NSUInteger threads_per_query = std::min((uint32_t)G, 32u);

        auto cmd = [impl_->queue commandBuffer];
        auto enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:impl_->pso_beam_search];
        [enc setBuffer:buf_dataset    offset:0 atIndex:0];
        [enc setBuffer:buf_graph      offset:0 atIndex:1];
        [enc setBuffer:buf_queries    offset:0 atIndex:2];
        [enc setBuffer:buf_out_nbrs   offset:0 atIndex:3];
        [enc setBuffer:buf_out_dists  offset:0 atIndex:4];
        [enc setBuffer:buf_cand_dists offset:0 atIndex:5];
        [enc setBuffer:buf_cand_nodes offset:0 atIndex:6];
        [enc setBuffer:buf_visited    offset:0 atIndex:7];
        uint32_t uN = (uint32_t)N, uD = (uint32_t)D, uG = G,
                 uK = (uint32_t)K;
        [enc setBytes:&uN length:4 atIndex:8];
        [enc setBytes:&uD length:4 atIndex:9];
        [enc setBytes:&uG length:4 atIndex:10];
        [enc setBytes:&uK length:4 atIndex:11];
        [enc setBytes:&beam_size length:4 atIndex:12];
        [enc setBytes:&max_iter  length:4 atIndex:13];

        // Entry nodes: Q uint32 values (provided or 0xFFFFFFFF = use hash)
        std::vector<uint32_t> entry_buf_data(static_cast<size_t>(Q),
                                              std::numeric_limits<uint32_t>::max());
        if (entry_nodes)
            std::copy_n(entry_nodes, static_cast<size_t>(Q), entry_buf_data.data());
        auto buf_entry = make_shared_buf(impl_->device, entry_buf_data.data(),
                              (NSUInteger)(Q * sizeof(uint32_t)));
        [enc setBuffer:buf_entry offset:0 atIndex:14];

        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)Q, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(threads_per_query, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error)
            throw std::runtime_error(cmd.error.localizedDescription.UTF8String);

        // Copy results back from shared MTLBuffers to caller's pointers
        std::copy_n((const uint32_t*)buf_out_nbrs.contents,
                    (size_t)(Q * K), out_nbrs);
        std::copy_n((const float*)buf_out_dists.contents,
                    (size_t)(Q * K), out_dists);
    }
    return 1;
}

} // namespace cuvs_silicon
