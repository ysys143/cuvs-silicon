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
    id<MTLComputePipelineState> pso_copy                  = nil;
    id<MTLComputePipelineState> pso_beam_search           = nil;
    id<MTLComputePipelineState> pso_beam_search_multi_cta = nil;
    id<MTLComputePipelineState> pso_nn_descent            = nil;
    id<MTLComputePipelineState> pso_random_bucketing      = nil;
    id<MTLComputePipelineState> pso_brute_force_topk      = nil;
    id<MTLComputePipelineState> pso_l2_topk_from_cross    = nil;
    bool                        available = false;
    std::string                 chip_model_str;

    // Dataset norm cache — invalidated when dataset pointer or shape changes.
    const float*       cached_dataset_ptr = nullptr;
    int64_t            cached_N = 0, cached_D = 0;
    std::vector<float> cached_dataset_norms;

    // Reusable cross-product buffer — avoids 4MB+ alloc per search call.
    std::vector<float> cross_buf;
    int64_t            cross_buf_Q = 0, cross_buf_N = 0;

    // Metal buffer cache for brute-force GPU search.
    id<MTLBuffer>   buf_dataset_metal    = nil;  // float32 (shared)
    id<MTLBuffer>   buf_dataset_fp16     = nil;  // float16 (private) — halves matmul reads
    id<MTLBuffer>   buf_d_norms_metal    = nil;
    const float*    buf_dataset_ptr      = nullptr;
    int64_t         buf_dataset_N        = 0, buf_dataset_D = 0;
    id<MTLBuffer>   buf_cross_metal      = nil;
    int64_t         buf_cross_Q          = 0, buf_cross_N   = 0;
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
        impl_->pso_copy                  = make_pso(@"copy_kernel");
        impl_->pso_beam_search           = make_pso(@"cagra_beam_search");
        impl_->pso_beam_search_multi_cta = make_pso(@"cagra_beam_search_multi_cta");
        impl_->pso_nn_descent            = make_pso(@"nn_descent");
        impl_->pso_random_bucketing      = make_pso(@"random_bucketing");
        impl_->pso_brute_force_topk      = make_pso(@"brute_force_topk");
        impl_->pso_l2_topk_from_cross    = make_pso(@"l2_topk_from_cross");

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

    // ── Two-pass Metal GPU brute-force ────────────────────────────────────
    // Pass 1 (MPS): cross[Q×N] = queries @ dataset^T  — dataset read ONCE
    // Pass 2 (kernel): dist = q_norm - 2*cross + d_norm → top-K per query
    // Transfer to CPU: only Q×K results (4KB). No Q×N (400MB) CPU transfer.
    @autoreleasepool {
        // Cache dataset + d_norms Metal buffers
        if (impl_->buf_dataset_ptr != dataset ||
            impl_->buf_dataset_N != N || impl_->buf_dataset_D != D) {
            // float32 shared buffer (for reference if needed)
            impl_->buf_dataset_metal = make_shared_buf(impl_->device, dataset,
                (NSUInteger)(N * D * sizeof(float)));
            // float16 private buffer — halves matmul memory reads (4GB→2GB)
            impl_->buf_dataset_fp16 = [impl_->device
                newBufferWithLength:(NSUInteger)(N * D * sizeof(__fp16))
                options:MTLResourceStorageModeShared];
            auto* fp16_ptr = reinterpret_cast<__fp16*>(impl_->buf_dataset_fp16.contents);
            for (int64_t i = 0; i < N * D; ++i)
                fp16_ptr[i] = (__fp16)dataset[i];
            impl_->cached_dataset_norms = compute_row_norms(dataset, N, D);
            impl_->buf_d_norms_metal = make_shared_buf(impl_->device,
                impl_->cached_dataset_norms.data(),
                (NSUInteger)(N * sizeof(float)));
            impl_->buf_dataset_ptr = dataset;
            impl_->buf_dataset_N = N; impl_->buf_dataset_D = D;
            impl_->cached_dataset_ptr = dataset;
            impl_->cached_N = N; impl_->cached_D = D;
        }

        // Cache cross buffer (Q×N, GPU-only — never transferred to CPU)
        if (impl_->buf_cross_Q != Q || impl_->buf_cross_N != N) {
            impl_->buf_cross_metal = [impl_->device
                newBufferWithLength:(NSUInteger)(Q * N * sizeof(float))
                options:MTLResourceStorageModePrivate];  // GPU-only, faster
            impl_->buf_cross_Q = Q; impl_->buf_cross_N = N;
        }

        auto buf_out_idx  = make_empty_buf(impl_->device,
                                (NSUInteger)(Q * K * sizeof(uint32_t)));
        auto buf_out_dist = make_empty_buf(impl_->device,
                                (NSUInteger)(Q * K * sizeof(float)));

        // float16 query buffer (Q×D × 2 bytes = tiny)
        auto buf_queries_fp16 = [impl_->device
            newBufferWithLength:(NSUInteger)(Q * D * sizeof(__fp16))
            options:MTLResourceStorageModeShared];
        auto* q16 = reinterpret_cast<__fp16*>(buf_queries_fp16.contents);
        for (int64_t i = 0; i < Q * D; ++i) q16[i] = (__fp16)queries[i];

        // Precompute query norms in float32 (precise)
        const auto q_norms_vec = compute_row_norms(queries, Q, D);
        auto buf_q_norms = make_shared_buf(impl_->device, q_norms_vec.data(),
                               (NSUInteger)(Q * sizeof(float)));

        // ── Pass 1: MPS float16×float16→float32 matmul ───────────────────
        MPSMatrixDescriptor* dQ = [MPSMatrixDescriptor
            matrixDescriptorWithRows:(NSUInteger)Q columns:(NSUInteger)D
            rowBytes:(NSUInteger)(D*sizeof(__fp16)) dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor* dD = [MPSMatrixDescriptor
            matrixDescriptorWithRows:(NSUInteger)N columns:(NSUInteger)D
            rowBytes:(NSUInteger)(D*sizeof(__fp16)) dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor* dC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:(NSUInteger)Q columns:(NSUInteger)N
            rowBytes:(NSUInteger)(N*sizeof(float)) dataType:MPSDataTypeFloat32];

        MPSMatrix* matQ = [[MPSMatrix alloc] initWithBuffer:buf_queries_fp16      descriptor:dQ];
        MPSMatrix* matD = [[MPSMatrix alloc] initWithBuffer:impl_->buf_dataset_fp16 descriptor:dD];
        MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:impl_->buf_cross_metal  descriptor:dC];

        MPSMatrixMultiplication* gemm = [[MPSMatrixMultiplication alloc]
            initWithDevice:impl_->device transposeLeft:NO transposeRight:YES
            resultRows:(NSUInteger)Q resultColumns:(NSUInteger)N
            interiorColumns:(NSUInteger)D alpha:1.0 beta:0.0];

        auto cmd1 = [impl_->queue commandBuffer];
        [gemm encodeToCommandBuffer:cmd1 leftMatrix:matQ rightMatrix:matD resultMatrix:matC];
        [cmd1 commit];
        [cmd1 waitUntilCompleted];

        // ── Pass 2: L2 distances + top-K on GPU ───────────────────────────
        uint32_t uN=(uint32_t)N, uK=(uint32_t)K;
        auto cmd2 = [impl_->queue commandBuffer];
        auto enc  = [cmd2 computeCommandEncoder];
        [enc setComputePipelineState:impl_->pso_l2_topk_from_cross];
        [enc setBuffer:impl_->buf_cross_metal   offset:0 atIndex:0];
        [enc setBuffer:buf_q_norms              offset:0 atIndex:1];
        [enc setBuffer:impl_->buf_d_norms_metal offset:0 atIndex:2];
        [enc setBuffer:buf_out_idx              offset:0 atIndex:3];
        [enc setBuffer:buf_out_dist             offset:0 atIndex:4];
        [enc setBytes:&uN length:4 atIndex:5];
        [enc setBytes:&uK length:4 atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)Q, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];
        [cmd2 commit];
        [cmd2 waitUntilCompleted];

        if (cmd2.error)
            throw std::runtime_error(cmd2.error.localizedDescription.UTF8String);

        // Only Q×K = 4KB transferred to CPU
        std::copy_n((const uint32_t*)buf_out_idx.contents,
                    static_cast<size_t>(Q * K), out_neighbors);
        std::copy_n((const float*)buf_out_dist.contents,
                    static_cast<size_t>(Q * K), out_distances);
    }
    return 1;
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

        fprintf(stderr, "  [build] K-means K=%lld  ", (long long)K);
        fflush(stderr);
        for (int iter = 0; iter < km_iters; ++iter) {
            fprintf(stderr, "%d ", iter+1); fflush(stderr);
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

        fprintf(stderr, "\n  [build] IVF seeding K=%lld  0%%", (long long)K);
        fflush(stderr);

        // ── IVF_PQ: Product Quantization encoding (replaces probe_vecs gather) ──
        // Reduces random-access gather from 16MB per cluster → 32KB.
        // PQ_M subspaces each with 256 centers, stored as uint8 codes.
        // LUT (2MB) fits in L3 cache; pq_codes (8MB) mostly in L3 too.
        constexpr int64_t PQ_M   = 8;        // subspaces (D/PQ_DIM)
        constexpr int64_t PQ_DIM = 128;      // dims per subspace (D/PQ_M = 1024/8)
        constexpr int     PQ_K   = 256;      // centers per subspace (8-bit codes)
        constexpr int     pq_iters = 10;     // K-means iterations for PQ training

        std::vector<float>   pq_centers(static_cast<size_t>(PQ_M * PQ_K * PQ_DIM));
        std::vector<uint8_t> pq_codes(static_cast<size_t>(N * PQ_M));

        // Use dataset directly with lda=D stride — no sub_vecs copy needed.
        // cblas_sgemm supports non-unit leading dimension for strided access.
        {
            std::vector<float>   sub_cross(static_cast<size_t>(chunk * PQ_K));
            std::vector<int32_t> sub_assign(static_cast<size_t>(N), 0);
            std::vector<float>   ctr_norms(static_cast<size_t>(PQ_K));
            std::vector<int64_t> counts(static_cast<size_t>(PQ_K), 0LL);

            for (int64_t m = 0; m < PQ_M; ++m) {
                const int64_t dim_off = m * PQ_DIM;
                float* ctr = pq_centers.data() + m * PQ_K * PQ_DIM;

                // Init centers: stride-based (read directly from dataset)
                {
                    const double stride = static_cast<double>(N) / PQ_K;
                    for (int c = 0; c < PQ_K; ++c) {
                        int64_t idx = static_cast<int64_t>(c * stride + stride * 0.5);
                        idx = std::min(idx, N - 1);
                        std::copy_n(dataset + idx*D + dim_off, PQ_DIM, ctr + c*PQ_DIM);
                    }
                }

                // K-means E-M: sgemm with lda=D reads subspace directly from dataset
                for (int iter = 0; iter < pq_iters; ++iter) {
                    for (int c = 0; c < PQ_K; ++c) {
                        float s = 0.f;
                        for (int d = 0; d < PQ_DIM; ++d) s += ctr[c*PQ_DIM+d]*ctr[c*PQ_DIM+d];
                        ctr_norms[static_cast<size_t>(c)] = s;
                    }
                    bool changed = (iter == 0);
                    for (int64_t start = 0; start < N; start += chunk) {
                        const int64_t Bs = std::min(chunk, N - start);
                        if (static_cast<int64_t>(sub_cross.size()) < Bs * PQ_K)
                            sub_cross.resize(static_cast<size_t>(Bs * PQ_K));
                        // lda=D: each row of dataset has D elements, we read PQ_DIM from offset
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                    (int)Bs, (int)PQ_K, (int)PQ_DIM,
                                    1.0f, dataset + start*D + dim_off, (int)D,
                                    ctr, (int)PQ_DIM,
                                    0.0f, sub_cross.data(), (int)PQ_K);
                        for (int64_t b = 0; b < Bs; ++b) {
                            const int64_t vi = start + b;
                            // norm of subspace slice
                            float vi_norm = 0.f;
                            const float* sv = dataset + vi*D + dim_off;
                            for (int d = 0; d < PQ_DIM; ++d) vi_norm += sv[d]*sv[d];
                            float best_d = std::numeric_limits<float>::infinity();
                            int32_t best_c = 0;
                            for (int c = 0; c < PQ_K; ++c) {
                                float dd = vi_norm - 2.f*sub_cross[b*PQ_K+c] + ctr_norms[c];
                                if (dd < best_d) { best_d = dd; best_c = c; }
                            }
                            if (sub_assign[vi] != best_c) { changed = true; sub_assign[vi] = best_c; }
                        }
                    }
                    if (!changed) break;
                    // M-step: accumulate directly from dataset with stride
                    std::fill(ctr, ctr + PQ_K * PQ_DIM, 0.f);
                    std::fill(counts.begin(), counts.end(), 0LL);
                    for (int64_t i = 0; i < N; ++i) {
                        int c = sub_assign[i];
                        counts[static_cast<size_t>(c)]++;
                        cblas_saxpy((int)PQ_DIM, 1.0f,
                                    dataset + i*D + dim_off, 1,
                                    ctr + c*PQ_DIM, 1);
                    }
                    for (int c = 0; c < PQ_K; ++c)
                        if (counts[static_cast<size_t>(c)] > 0)
                            cblas_sscal((int)PQ_DIM, 1.0f/counts[static_cast<size_t>(c)],
                                        ctr + c*PQ_DIM, 1);
                }

                // Encoding: one final E-step via sgemm (same as K-means E-step)
                for (int64_t start = 0; start < N; start += chunk) {
                    const int64_t Bs = std::min(chunk, N - start);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                (int)Bs, (int)PQ_K, (int)PQ_DIM,
                                1.0f, dataset + start*D + dim_off, (int)D,
                                ctr, (int)PQ_DIM,
                                0.0f, sub_cross.data(), (int)PQ_K);
                    for (int64_t b = 0; b < Bs; ++b) {
                        const int64_t vi = start + b;
                        const float* sv = dataset + vi*D + dim_off;
                        float vi_norm = 0.f;
                        for (int d = 0; d < PQ_DIM; ++d) vi_norm += sv[d]*sv[d];
                        float best_d = std::numeric_limits<float>::infinity();
                        uint8_t best_c = 0;
                        for (int c = 0; c < PQ_K; ++c) {
                            float dd = vi_norm - 2.f*sub_cross[b*PQ_K+c] + ctr_norms[c];
                            if (dd < best_d) { best_d = dd; best_c = (uint8_t)c; }
                        }
                        pq_codes[vi * PQ_M + m] = best_c;
                    }
                }
            }
        }

        // ── Precompute symmetric PQ LUT: lut[m×K×K + a×K + b] = dist(ctr[m][a], ctr[m][b])
        // Size: PQ_M × 256 × 256 × 4 bytes = 8 × 256KB = 2MB → fits in L3 cache.
        std::vector<float> pq_lut(static_cast<size_t>(PQ_M * PQ_K * PQ_K));
        for (int64_t m = 0; m < PQ_M; ++m) {
            const float* ctr_m = pq_centers.data() + m * PQ_K * PQ_DIM;
            for (int a = 0; a < PQ_K; ++a) {
                for (int b = a; b < PQ_K; ++b) {
                    float d2 = 0.f;
                    const float* ca = ctr_m + a*PQ_DIM, *cb = ctr_m + b*PQ_DIM;
                    for (int d = 0; d < PQ_DIM; ++d) { float dif = ca[d]-cb[d]; d2 += dif*dif; }
                    pq_lut[m*PQ_K*PQ_K + a*PQ_K + b] = d2;
                    pq_lut[m*PQ_K*PQ_K + b*PQ_K + a] = d2;
                }
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
            if (k % (K / 10 + 1) == 0) {
                fprintf(stderr, "\r  [build] IVF seeding  %3lld%%",
                        (long long)(k * 100 / K));
                fflush(stderr);
            }
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

            // IVF_PQ: use PQ LUT distances instead of sgemm.
            // Eliminates probe_vecs gather (16MB random reads → 32KB PQ code reads).
            // PQ distance is approximate; exact distance computed only for top-G.
            constexpr int64_t MAX_BATCH = 256;
            std::vector<float>    pq_cross(static_cast<size_t>(MAX_BATCH * M));
            std::vector<uint32_t> midx(static_cast<size_t>(M));

            for (int64_t istart = 0; istart < nk; istart += MAX_BATCH) {
                const int64_t iend = std::min(istart + MAX_BATCH, nk);
                const int64_t ibsz = iend - istart;

                // PQ distance lookup: O(ibsz × M × PQ_M) byte reads from L3-cached arrays
                for (int64_t ii = 0; ii < ibsz; ++ii) {
                    const uint32_t vi = own_full[static_cast<size_t>(istart + ii)];
                    const uint8_t* vi_c = pq_codes.data() + (size_t)vi * PQ_M;
                    for (int64_t pm = 0; pm < M; ++pm) {
                        const uint8_t* pj_c = pq_codes.data() +
                            (size_t)probe[static_cast<size_t>(pm)] * PQ_M;
                        float d = 0.f;
                        for (int64_t sm = 0; sm < PQ_M; ++sm)
                            d += pq_lut[sm*PQ_K*PQ_K + vi_c[sm]*PQ_K + pj_c[sm]];
                        pq_cross[ii * M + pm] = d;
                    }
                }

                // Sort by PQ-approximate distance, then compute exact distance for top-G
                const int64_t take = std::min((int64_t)G, M-1);
                for (int64_t i = istart; i < iend; ++i) {
                    const uint32_t vi = own_full[static_cast<size_t>(i)];
                    const float ni = norms[static_cast<size_t>(vi)];
                    const float* cr = pq_cross.data() + (i-istart)*M;
                    std::iota(midx.begin(), midx.end(), 0u);
                    std::partial_sort(midx.begin(), midx.begin()+take, midx.end(),
                        [&](uint32_t a, uint32_t b) {
                            if (probe[a] == vi) return false;
                            if (probe[b] == vi) return true;
                            return cr[a] < cr[b];  // PQ approx dist for ranking
                        });
                    // Exact distance only for top-G candidates (O(G × D), not O(M × D))
                    const float* vi_v = dataset + (size_t)vi * D;
                    for (int64_t t = 0; t < take; ++t) {
                        uint32_t nbr = probe[midx[static_cast<size_t>(t)]];
                        if (nbr == vi) continue;
                        const float* nbr_v = dataset + (size_t)nbr * D;
                        float d = 0.f;
                        for (int64_t dd = 0; dd < D; ++dd) {
                            float dif = vi_v[dd] - nbr_v[dd]; d += dif*dif;
                        }
                        insert_neighbor(vi, nbr, d);
                    }
                }
            }
        }
    } // end IVF seeding
    fprintf(stderr, "\r  [build] IVF seeding  100%%\n"); fflush(stderr);

    // ── Phase 1b: Random bucketing pass (GPU-accelerated) ─────────────────
    // IVF seeding lacks cross-cluster edges. One random bucketing pass adds
    // global diversity. Metal kernel: 1 threadgroup per bucket, 1 thread per
    // vector — 391 buckets dispatched in parallel on GPU.
    if (N > FULL_SEEDING_LIMIT && impl_->pso_random_bucketing) {
        constexpr uint32_t rbsz = 256;

        // Fisher-Yates shuffle on CPU (O(N), ~1ms)
        std::vector<uint32_t> rperm(static_cast<size_t>(N));
        std::iota(rperm.begin(), rperm.end(), 0u);
        uint64_t rrng = 0xFEDCBA9876543210ULL;
        for (int64_t i = N-1; i > 0; --i) {
            rrng = rrng * 6364136223846793005ULL + 1442695040888963407ULL;
            int64_t j = (int64_t)(rrng >> 33) % (i+1);
            std::swap(rperm[static_cast<size_t>(i)], rperm[static_cast<size_t>(j)]);
        }

        @autoreleasepool {
            const NSUInteger n_buckets = (static_cast<NSUInteger>(N) + rbsz - 1) / rbsz;

            auto buf_dataset = make_shared_buf(impl_->device, dataset,
                                   (NSUInteger)(N * D * sizeof(float)));
            auto buf_rperm   = make_shared_buf(impl_->device, rperm.data(),
                                   (NSUInteger)(N * sizeof(uint32_t)));
            auto buf_graph   = [impl_->device
                newBufferWithBytes:graph.data()
                length:(NSUInteger)(N * G * sizeof(uint32_t))
                options:MTLResourceStorageModeShared];
            auto buf_gdist   = [impl_->device
                newBufferWithBytes:graph_dist.data()
                length:(NSUInteger)(N * G * sizeof(float))
                options:MTLResourceStorageModeShared];

            uint32_t uN = (uint32_t)N, uD = (uint32_t)D, uG = G;
            auto cmd = [impl_->queue commandBuffer];
            auto enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:impl_->pso_random_bucketing];
            [enc setBuffer:buf_dataset offset:0 atIndex:0];
            [enc setBuffer:buf_rperm   offset:0 atIndex:1];
            [enc setBuffer:buf_graph   offset:0 atIndex:2];
            [enc setBuffer:buf_gdist   offset:0 atIndex:3];
            [enc setBytes:&uN   length:4 atIndex:4];
            [enc setBytes:&uD   length:4 atIndex:5];
            [enc setBytes:&uG   length:4 atIndex:6];
            [enc setBytes:&rbsz length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(n_buckets, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(rbsz, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            std::copy_n((const uint32_t*)buf_graph.contents,
                        static_cast<size_t>(N * G), graph.data());
            std::copy_n((const float*)buf_gdist.contents,
                        static_cast<size_t>(N * G), graph_dist.data());
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

            fprintf(stderr, "  [build] nn-descent  "); fflush(stderr);
            for (uint32_t iter = 0; iter < n_descent_iters; ++iter) {
                fprintf(stderr, "%u/%u ", iter+1, n_descent_iters); fflush(stderr);
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

            fprintf(stderr, "\n"); fflush(stderr);

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

    // Multi-CTA kernel available (cagra_beam_search_multi_cta) but disabled:
    // random-access memory-bound workload means teams compete for cache,
    // causing worse performance than single-CTA (92ms vs 54ms at Q=1).
    // Kept in codebase for future experimentation on different hardware.
    const uint32_t N_TEAMS = 1u;

    if (N_TEAMS > 1) {
        @autoreleasepool {
            const uint32_t visited_words = (static_cast<uint32_t>(N) + 31u) / 32u;
            // Multi-CTA uses smaller beam for speed; diversity compensates quality.
            const uint32_t mc_beam = std::min(beam_size, 32u);

            auto buf_dataset = make_shared_buf(impl_->device, dataset,
                                   (NSUInteger)(N * D * sizeof(float)));
            auto buf_graph   = make_shared_buf(impl_->device, knn_graph,
                                   (NSUInteger)((uint64_t)N * G * sizeof(uint32_t)));
            auto buf_queries = make_shared_buf(impl_->device, queries,
                                   (NSUInteger)(Q * D * sizeof(float)));

            // per-team candidate buffers: Q × N_TEAMS × mc_beam
            const size_t team_buf_sz = (size_t)Q * N_TEAMS * mc_beam;
            auto buf_team_dists = make_empty_buf(impl_->device,
                                      (NSUInteger)(team_buf_sz * 4u));
            auto buf_team_nodes = make_empty_buf(impl_->device,
                                      (NSUInteger)(team_buf_sz * 4u));
            // per-team visited: Q × N_TEAMS × visited_words
            auto buf_visited = make_empty_buf(impl_->device,
                                   (NSUInteger)((uint64_t)Q * N_TEAMS * visited_words * 4u));
            memset(buf_visited.contents, 0,
                   (size_t)((uint64_t)Q * N_TEAMS * visited_words * 4u));

            // Entry nodes buffer
            std::vector<uint32_t> entry_buf_data(static_cast<size_t>(Q),
                                                  std::numeric_limits<uint32_t>::max());
            if (entry_nodes)
                std::copy_n(entry_nodes, static_cast<size_t>(Q), entry_buf_data.data());
            auto buf_entry = make_shared_buf(impl_->device, entry_buf_data.data(),
                                  (NSUInteger)(Q * sizeof(uint32_t)));

            uint32_t uN=(uint32_t)N, uD=(uint32_t)D, uG=G, uK=(uint32_t)K,
                     uNT=N_TEAMS;
            const NSUInteger tpq = std::min((uint32_t)G, 32u);

            auto cmd = [impl_->queue commandBuffer];
            auto enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:impl_->pso_beam_search_multi_cta];
            [enc setBuffer:buf_dataset    offset:0 atIndex:0];
            [enc setBuffer:buf_graph      offset:0 atIndex:1];
            [enc setBuffer:buf_queries    offset:0 atIndex:2];
            [enc setBuffer:buf_team_dists offset:0 atIndex:3];
            [enc setBuffer:buf_team_nodes offset:0 atIndex:4];
            [enc setBuffer:buf_visited    offset:0 atIndex:5];
            [enc setBytes:&uN       length:4 atIndex:6];
            [enc setBytes:&uD       length:4 atIndex:7];
            [enc setBytes:&uG       length:4 atIndex:8];
            [enc setBytes:&uK       length:4 atIndex:9];
            [enc setBytes:&mc_beam  length:4 atIndex:10];
            [enc setBytes:&max_iter length:4 atIndex:11];
            [enc setBytes:&uNT      length:4 atIndex:12];
            [enc setBuffer:buf_entry offset:0 atIndex:13];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)Q, N_TEAMS, 1)
                threadsPerThreadgroup:MTLSizeMake(tpq, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            if (cmd.error)
                throw std::runtime_error(cmd.error.localizedDescription.UTF8String);

            // ── CPU merge: N_TEAMS × mc_beam candidates → top-K per query ──
            const auto* td = static_cast<const float*>(buf_team_dists.contents);
            const auto* tn = static_cast<const uint32_t*>(buf_team_nodes.contents);
            for (int64_t q = 0; q < Q; ++q) {
                // Collect all valid candidates from all teams
                std::vector<std::pair<float, uint32_t>> cands;
                cands.reserve(N_TEAMS * mc_beam);
                for (uint32_t t = 0; t < N_TEAMS; ++t) {
                    const size_t base = ((size_t)q * N_TEAMS + t) * mc_beam;
                    for (uint32_t b = 0; b < mc_beam; ++b) {
                        uint32_t node = tn[base + b];
                        if (node == 0xFFFFFFFFu) continue;
                        float d = td[base + b];
                        float absd = d < 0.f ? (-d - 1.f) : d;
                        cands.emplace_back(absd, node);
                    }
                }
                // Deduplicate and partial-sort for top-K
                std::sort(cands.begin(), cands.end());
                cands.erase(std::unique(cands.begin(), cands.end(),
                    [](const auto& a, const auto& b){ return a.second == b.second; }),
                    cands.end());
                const int64_t take = std::min((int64_t)K, (int64_t)cands.size());
                for (int64_t k = 0; k < K; ++k) {
                    if (k < take) {
                        out_nbrs [q * K + k] = cands[static_cast<size_t>(k)].second;
                        out_dists[q * K + k] = cands[static_cast<size_t>(k)].first;
                    } else {
                        out_nbrs [q * K + k] = 0xFFFFFFFFu;
                        out_dists[q * K + k] = std::numeric_limits<float>::infinity();
                    }
                }
            }
        }
        return 1;
    }

    // ── Single-CTA path (Q > 16 or multi-CTA unavailable) ────────────
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
        auto buf_cand_dists = make_empty_buf(impl_->device,
                               (NSUInteger)((uint64_t)Q * beam_size * 4u));
        auto buf_cand_nodes = make_empty_buf(impl_->device,
                               (NSUInteger)((uint64_t)Q * beam_size * 4u));
        auto buf_visited = make_empty_buf(impl_->device,
                               (NSUInteger)((uint64_t)Q * visited_words * 4u));
        memset(buf_visited.contents, 0,
               (size_t)((uint64_t)Q * visited_words * 4u));

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

        std::copy_n((const uint32_t*)buf_out_nbrs.contents,
                    (size_t)(Q * K), out_nbrs);
        std::copy_n((const float*)buf_out_dists.contents,
                    (size_t)(Q * K), out_dists);
    }
    return 1;
}

} // namespace cuvs_silicon
