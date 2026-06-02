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

    // ── Phase 1: Seeding ──────────────────────────────────────────────────
    // Small N: exact N×N seeding via cblas_sgemm.
    // Large N: IVF K-means seeding — K=sqrt(N) clusters, probe own + n_probe
    //          nearest clusters → high-quality graph without N² cost.

    constexpr int64_t FULL_SEEDING_LIMIT = 50000;

    std::vector<uint32_t> graph(static_cast<size_t>(N) * G,
                                 std::numeric_limits<uint32_t>::max());
    std::vector<float> graph_dist(static_cast<size_t>(N) * G,
                                   std::numeric_limits<float>::infinity());

    std::vector<float> norms(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i) {
        float s = 0.f; const float* v = dataset + i * D;
        for (int64_t d = 0; d < D; ++d) s += v[d] * v[d];
        norms[static_cast<size_t>(i)] = s;
    }

    auto insert_neighbor = [&](int64_t vi, uint32_t nbr, float dist) {
        const size_t row = static_cast<size_t>(vi) * G;
        float worst = graph_dist[row]; uint32_t worst_g = 0;
        for (uint32_t g = 1; g < G; ++g) {
            if (graph_dist[row+g] > worst) { worst = graph_dist[row+g]; worst_g = g; }
        }
        if (dist < worst) { graph[row+worst_g] = nbr; graph_dist[row+worst_g] = dist; }
    };

    if (N <= FULL_SEEDING_LIMIT) {
        // Exact N×N seeding via cblas_sgemm (AMX)
        std::vector<float> cross(static_cast<size_t>(N * N));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)N, (int)N, (int)D,
                    1.0f, dataset, (int)D, dataset, (int)D,
                    0.0f, cross.data(), (int)N);
        std::vector<uint32_t> idx_buf(static_cast<size_t>(N));
        for (int64_t i = 0; i < N; ++i) {
            const float ni = norms[static_cast<size_t>(i)];
            std::iota(idx_buf.begin(), idx_buf.end(), 0u);
            idx_buf[static_cast<size_t>(i)] = idx_buf[static_cast<size_t>(N-1)];
            const int64_t take = std::min((int64_t)G, N-1);
            std::partial_sort(idx_buf.begin(), idx_buf.begin()+take,
                              idx_buf.begin()+N-1,
                [&](uint32_t a, uint32_t b) {
                    float da = ni - 2.f*cross[i*N+a] + norms[a];
                    float db = ni - 2.f*cross[i*N+b] + norms[b];
                    return da < db;
                });
            for (int64_t k = 0; k < take; ++k) {
                uint32_t nbr = idx_buf[static_cast<size_t>(k)];
                float d = ni - 2.f*cross[i*N+nbr] + norms[nbr];
                insert_neighbor(i, nbr, d);
            }
        }

    } else {
        // IVF K-means seeding for large N
        const int64_t K = std::min<int64_t>(
            std::max<int64_t>(static_cast<int64_t>(std::sqrt(static_cast<double>(N))), G+1),
            2000LL);
        constexpr int km_iters = 20;
        constexpr int n_probe  = 3;   // probe own + 3 nearest clusters
        constexpr int64_t chunk = 4096; // chunk size for E-step to limit memory

        // ── K-means initialization: evenly-spaced strides ────────────────
        // Random LCG can repeat indices (39% collision chance for K=316, N=100K)
        // causing degenerate clusters. Stride-based spacing guarantees K distinct
        // centroids spread evenly across the dataset.
        std::vector<float> centroids(static_cast<size_t>(K * D));
        {
            const double stride = static_cast<double>(N) / K;
            for (int64_t k = 0; k < K; ++k) {
                int64_t idx = static_cast<int64_t>(k * stride + stride * 0.5);
                idx = std::min(idx, N - 1);
                std::copy_n(dataset + idx*D, static_cast<size_t>(D),
                            centroids.data() + k*D);
            }
        }

        std::vector<uint32_t> assignments(static_cast<size_t>(N), 0u);
        std::vector<float> c_norms(static_cast<size_t>(K));
        std::vector<float> cross_ck(static_cast<size_t>(chunk * K));

        for (int iter = 0; iter < km_iters; ++iter) {
            // centroid norms
            for (int64_t k = 0; k < K; ++k) {
                float s = 0.f; const float* c = centroids.data() + k*D;
                for (int64_t d = 0; d < D; ++d) s += c[d]*c[d];
                c_norms[static_cast<size_t>(k)] = s;
            }
            // E-step: chunked cblas_sgemm → argmin
            bool changed = (iter == 0);
            for (int64_t start = 0; start < N; start += chunk) {
                const int64_t Bs = std::min(chunk, N - start);
                if (static_cast<int64_t>(cross_ck.size()) < Bs * K)
                    cross_ck.resize(static_cast<size_t>(Bs * K));
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            (int)Bs, (int)K, (int)D,
                            1.0f, dataset + start*D, (int)D,
                            centroids.data(), (int)D,
                            0.0f, cross_ck.data(), (int)K);
                for (int64_t b = 0; b < Bs; ++b) {
                    const int64_t vi = start + b;
                    const float ni = norms[static_cast<size_t>(vi)];
                    float best_d = std::numeric_limits<float>::infinity();
                    uint32_t best_k = 0;
                    for (int64_t k = 0; k < K; ++k) {
                        float d = ni - 2.f*cross_ck[b*K+k] + c_norms[static_cast<size_t>(k)];
                        if (d < best_d) { best_d = d; best_k = static_cast<uint32_t>(k); }
                    }
                    if (assignments[static_cast<size_t>(vi)] != best_k) {
                        changed = true; assignments[static_cast<size_t>(vi)] = best_k;
                    }
                }
            }
            if (!changed) break;
            // M-step: update centroids
            std::fill(centroids.begin(), centroids.end(), 0.f);
            std::vector<int64_t> counts(static_cast<size_t>(K), 0LL);
            for (int64_t i = 0; i < N; ++i) {
                const uint32_t k = assignments[static_cast<size_t>(i)];
                counts[static_cast<size_t>(k)]++;
                cblas_saxpy((int)D, 1.0f, dataset + i*D, 1,
                            centroids.data() + k*D, 1);
            }
            for (int64_t k = 0; k < K; ++k) {
                if (counts[static_cast<size_t>(k)] > 0)
                    cblas_sscal((int)D, 1.0f/counts[static_cast<size_t>(k)],
                                centroids.data() + k*D, 1);
            }
        }

        // ── Build cluster membership lists ─────────────────────────────
        std::vector<std::vector<uint32_t>> clusters(static_cast<size_t>(K));
        for (int64_t i = 0; i < N; ++i)
            clusters[assignments[static_cast<size_t>(i)]].push_back(
                static_cast<uint32_t>(i));

        // ── Find n_probe nearest clusters for each cluster (centroid dist) ──
        // centroid-centroid cross[K×K]
        std::vector<float> cc_cross(static_cast<size_t>(K * K));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)K, (int)K, (int)D,
                    1.0f, centroids.data(), (int)D, centroids.data(), (int)D,
                    0.0f, cc_cross.data(), (int)K);
        // reuse c_norms already computed above
        for (int64_t k = 0; k < K; ++k) {
            float s = 0.f; const float* c = centroids.data() + k*D;
            for (int64_t d = 0; d < D; ++d) s += c[d]*c[d];
            c_norms[static_cast<size_t>(k)] = s;
        }
        std::vector<std::vector<uint32_t>> near_clusters(static_cast<size_t>(K));
        {
            std::vector<uint32_t> kidx(static_cast<size_t>(K));
            for (int64_t k = 0; k < K; ++k) {
                std::iota(kidx.begin(), kidx.end(), 0u);
                kidx[static_cast<size_t>(k)] = kidx[static_cast<size_t>(K-1)];
                const int64_t take = std::min((int64_t)n_probe, K-1);
                const float ck = c_norms[static_cast<size_t>(k)];
                std::partial_sort(kidx.begin(), kidx.begin()+take, kidx.begin()+K-1,
                    [&](uint32_t a, uint32_t b) {
                        float da = ck - 2.f*cc_cross[k*K+a] + c_norms[a];
                        float db = ck - 2.f*cc_cross[k*K+b] + c_norms[b];
                        return da < db;
                    });
                near_clusters[static_cast<size_t>(k)].assign(
                    kidx.begin(), kidx.begin()+take);
            }
        }

        // ── For each cluster, probe own + near clusters, compute k-NN ──
        // Cap own to max_own = N/K*4 to handle mildly unbalanced clusters.
        // With evenly-spaced init, most clusters are ~N/K in size; the cap
        // prevents the rare oversized cluster from causing OOM.
        const int64_t max_own = (N / K) * 4 + 64;

        for (int64_t k = 0; k < K; ++k) {
            const auto& own_full = clusters[static_cast<size_t>(k)];
            if (own_full.empty()) continue;

            // Sub-sample own if oversized
            const int64_t nk = std::min(static_cast<int64_t>(own_full.size()), max_own);

            // Build probe set: own (capped) + near clusters
            std::vector<uint32_t> probe;
            probe.reserve(static_cast<size_t>(nk +
                static_cast<int64_t>(n_probe) * (N / K + 1)));
            for (int64_t i = 0; i < nk; ++i)
                probe.push_back(own_full[static_cast<size_t>(i)]);
            for (auto nc : near_clusters[static_cast<size_t>(k)])
                for (auto v : clusters[static_cast<size_t>(nc)])
                    probe.push_back(v);

            // Cap probe to prevent sgemm blowup on unbalanced near clusters.
            constexpr int64_t MAX_PROBE = 8192;
            if (static_cast<int64_t>(probe.size()) > MAX_PROBE)
                probe.resize(static_cast<size_t>(MAX_PROBE));

            const int64_t M = static_cast<int64_t>(probe.size());
            if (M <= 1) continue;

            // Gather probe vectors and compute distances
            std::vector<float> probe_vecs(static_cast<size_t>(M * D));
            std::vector<float> probe_norms(static_cast<size_t>(M));
            for (int64_t m = 0; m < M; ++m) {
                uint32_t pv = probe[static_cast<size_t>(m)];
                std::copy_n(dataset + pv*D, static_cast<size_t>(D),
                            probe_vecs.data() + m*D);
                probe_norms[static_cast<size_t>(m)] = norms[static_cast<size_t>(pv)];
            }
            std::vector<float> own_vecs(static_cast<size_t>(nk * D));
            for (int64_t i = 0; i < nk; ++i)
                std::copy_n(dataset + own_full[static_cast<size_t>(i)]*D,
                            static_cast<size_t>(D), own_vecs.data() + i*D);

            // cross[nk×M] = own_vecs @ probe_vecs^T  (AMX)
            std::vector<float> cross_nm(static_cast<size_t>(nk * M));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        (int)nk, (int)M, (int)D,
                        1.0f, own_vecs.data(), (int)D,
                        probe_vecs.data(), (int)D,
                        0.0f, cross_nm.data(), (int)M);

            std::vector<uint32_t> midx(static_cast<size_t>(M));
            for (int64_t i = 0; i < nk; ++i) {
                const uint32_t vi = own_full[static_cast<size_t>(i)];
                const float ni = norms[static_cast<size_t>(vi)];
                const float* cr = cross_nm.data() + i*M;
                std::iota(midx.begin(), midx.end(), 0u);
                const int64_t take = std::min((int64_t)G, M-1);
                std::partial_sort(midx.begin(), midx.begin()+take, midx.end(),
                    [&](uint32_t a, uint32_t b) {
                        if (probe[a] == vi) return false;
                        if (probe[b] == vi) return true;
                        return ni-2.f*cr[a]+probe_norms[a] <
                               ni-2.f*cr[b]+probe_norms[b];
                    });
                for (int64_t t = 0; t < take; ++t) {
                    uint32_t nbr = probe[midx[static_cast<size_t>(t)]];
                    if (nbr == vi) continue;
                    float d = ni - 2.f*cr[midx[static_cast<size_t>(t)]]
                              + probe_norms[midx[static_cast<size_t>(t)]];
                    insert_neighbor(vi, nbr, d);
                }
            }
        }
    } // end IVF seeding

    // ── Phase 1b: Random bucketing pass (large N only) ─────────────────
    // IVF seeding produces locally well-connected clusters but lacks
    // cross-cluster edges. One random bucketing pass adds global diversity
    // (bridges between unrelated clusters) that nn-descent needs to converge.
    if (N > FULL_SEEDING_LIMIT) {
        const int64_t rbsz = 256;  // 256×256×4=256KB bcross fits in L2 cache
        std::vector<int64_t> rperm(static_cast<size_t>(N));
        std::iota(rperm.begin(), rperm.end(), 0LL);
        uint64_t rrng = 0xFEDCBA9876543210ULL;
        for (int64_t i = N-1; i > 0; --i) {
            rrng = rrng * 6364136223846793005ULL + 1442695040888963407ULL;
            int64_t j = (int64_t)(rrng >> 33) % (i+1);
            std::swap(rperm[static_cast<size_t>(i)], rperm[static_cast<size_t>(j)]);
        }
        for (int64_t start = 0; start < N; start += rbsz) {
            const int64_t end = std::min(start + rbsz, N);
            const int64_t Bs  = end - start;
            std::vector<float> bdata(static_cast<size_t>(Bs * D));
            for (int64_t b = 0; b < Bs; ++b)
                std::copy_n(dataset + rperm[static_cast<size_t>(start+b)]*D,
                            static_cast<size_t>(D), bdata.data() + b*D);
            std::vector<float> bcross(static_cast<size_t>(Bs * Bs));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        (int)Bs, (int)Bs, (int)D,
                        1.0f, bdata.data(), (int)D, bdata.data(), (int)D,
                        0.0f, bcross.data(), (int)Bs);
            std::vector<uint32_t> bidx(static_cast<size_t>(Bs));
            for (int64_t b = 0; b < Bs; ++b) {
                const int64_t vi = rperm[static_cast<size_t>(start+b)];
                const float ni  = norms[static_cast<size_t>(vi)];
                const float* cr = bcross.data() + b*Bs;
                std::iota(bidx.begin(), bidx.end(), 0u);
                bidx[static_cast<size_t>(b)] = bidx[static_cast<size_t>(Bs-1)];
                const int64_t take = std::min((int64_t)G, Bs-1);
                std::partial_sort(bidx.begin(), bidx.begin()+take,
                                  bidx.begin()+Bs-1,
                    [&](uint32_t a, uint32_t bb) {
                        const int64_t ga = rperm[static_cast<size_t>(start+a)];
                        const int64_t gb = rperm[static_cast<size_t>(start+bb)];
                        return ni-2.f*cr[a]+norms[static_cast<size_t>(ga)] <
                               ni-2.f*cr[bb]+norms[static_cast<size_t>(gb)];
                    });
                for (int64_t t = 0; t < take; ++t) {
                    uint32_t m = bidx[static_cast<size_t>(t)];
                    uint32_t nbr = static_cast<uint32_t>(
                        rperm[static_cast<size_t>(start+m)]);
                    float d = ni-2.f*cr[m]+norms[static_cast<size_t>(nbr)];
                    insert_neighbor(vi, nbr, d);
                }
            }
        }
    }

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
