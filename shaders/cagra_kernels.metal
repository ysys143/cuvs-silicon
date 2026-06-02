#include <metal_stdlib>
using namespace metal;

// P3: trivial dispatch verifier — input copy to output.
// Used only in test_metal_copy_kernel to confirm the full pipeline compiles
// and dispatches before writing the real search kernel.
kernel void copy_kernel(
    device const float* src [[ buffer(0) ]],
    device float*       dst [[ buffer(1) ]],
    uint gid [[ thread_position_in_grid ]])
{
    dst[gid] = src[gid];
}

// CAGRA beam search
// One threadgroup per query. Threads within a group process G neighbors in parallel.
// Working buffers (global memory, per query): candidates[], visited[] bitfield.
// Threadgroup shared memory: best_node, best_dist for broadcast.

struct BeamEntry {
    float dist;
    uint  node;
};

// cagra_beam_search: no global visited bitfield.
// Duplicate prevention is done by scanning the beam candidates (O(beam_size)).
// This allows nodes excluded at an early stage to be reconsidered later.
kernel void cagra_beam_search(
    device const float*  dataset    [[ buffer(0) ]],  // N×D
    device const uint*   graph      [[ buffer(1) ]],  // N×G
    device const float*  queries    [[ buffer(2) ]],  // Q×D
    device uint*         out_nbrs   [[ buffer(3) ]],  // Q×K output
    device float*        out_dists  [[ buffer(4) ]],  // Q×K output
    device float*        cand_dists [[ buffer(5) ]],  // Q×beam_size
    device uint*         cand_nodes [[ buffer(6) ]],  // Q×beam_size
    device uint*         visited    [[ buffer(7) ]],  // Q×ceil(N/32) bitfield
    constant uint&       N          [[ buffer(8) ]],
    constant uint&       D          [[ buffer(9) ]],
    constant uint&       G          [[ buffer(10) ]],
    constant uint&       K          [[ buffer(11) ]],
    constant uint&       beam_size   [[ buffer(12) ]],
    constant uint&       max_iter    [[ buffer(13) ]],
    device const uint*   entry_nodes [[ buffer(14) ]],  // Q entry points, or 0xFFFF if unused
    uint q_id                       [[ threadgroup_position_in_grid ]],
    uint t_id                       [[ thread_position_in_threadgroup ]],
    uint n_threads                  [[ threads_per_threadgroup ]])
{
    // Per-query base offsets
    const ulong qD  = (ulong)q_id * D;
    const ulong qBS = (ulong)q_id * beam_size;
    const ulong qK  = (ulong)q_id * K;
    const uint  vw  = (N + 31u) / 32u;
    const ulong qVW = (ulong)q_id * vw;

    // Threadgroup shared state
    threadgroup uint  best_node[1];
    threadgroup float best_dist_tg[1];
    // Phase-1 buffers: each thread writes one result per neighbor slot
    threadgroup float tg_dist[128];   // capacity = max G
    threadgroup uint  tg_node[128];

    // ── Init: thread 0 sets seed candidate ───────────────────────────
    if (t_id == 0) {
        // Use provided entry node if available, otherwise fall back to hash
        const uint seed = (entry_nodes[q_id] < N)
                          ? entry_nodes[q_id]
                          : (q_id * 2654435761u) % N;
        float d = 0.f;
        for (uint i = 0; i < D; ++i) {
            float delta = queries[qD + i] - dataset[(ulong)seed * D + i];
            d += delta * delta;
        }
        cand_dists[qBS + 0] = d;
        cand_nodes[qBS + 0] = seed;
        for (uint i = 1; i < beam_size; ++i) {
            cand_dists[qBS + i] = INFINITY;
            cand_nodes[qBS + i] = 0xFFFFFFFFu;
        }
        visited[qVW + (seed >> 5u)] |= (1u << (seed & 31u));
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ── Beam search loop ──────────────────────────────────────────────
    for (uint iter = 0; iter < max_iter; ++iter) {

        // Thread 0: find best non-expanded candidate, mark it expanded
        if (t_id == 0) {
            float bd  = INFINITY;
            uint  bn  = 0xFFFFFFFFu;
            for (uint i = 0; i < beam_size; ++i) {
                float cd = cand_dists[qBS + i];
                if (cd >= 0.f && cd < bd) {  // >=0 means not yet expanded
                    bd = cd; bn = cand_nodes[qBS + i];
                }
            }
            if (bn != 0xFFFFFFFFu) {
                // Mark expanded by negating distance
                for (uint i = 0; i < beam_size; ++i) {
                    if (cand_nodes[qBS + i] == bn) {
                        cand_dists[qBS + i] = -cand_dists[qBS + i] - 1.f;
                        break;
                    }
                }
            }
            best_node[0] = bn;
            best_dist_tg[0] = bd;
        }
        threadgroup_barrier(mem_flags::mem_device);

        if (best_node[0] == 0xFFFFFFFFu) break;

        const uint  expand = best_node[0];
        const ulong gBase  = (ulong)expand * G;

        // Phase 1 (parallel): each thread computes distances for its neighbors.
        // No visited check here — avoids write races on the bitfield.
        for (uint g = t_id; g < G; g += n_threads) {
            const uint nbr = graph[gBase + g];
            if (nbr < N) {
                float d = 0.f;
                const ulong nB = (ulong)nbr * D;
                for (uint dd = 0; dd < D; ++dd) {
                    float delta = queries[qD + dd] - dataset[nB + dd];
                    d += delta * delta;
                }
                tg_dist[g] = d;
                tg_node[g] = nbr;
            } else {
                tg_dist[g] = INFINITY;
                tg_node[g] = 0xFFFFFFFFu;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2 (thread 0 only): visited check + beam insertion.
        // Serial insertion eliminates all data races.
        if (t_id == 0) {
            for (uint g = 0; g < G; ++g) {
                const uint nbr = tg_node[g];
                if (nbr == 0xFFFFFFFFu) continue;
                const uint wrd = nbr >> 5u;
                const uint bit = 1u << (nbr & 31u);
                if (visited[qVW + wrd] & bit) continue;
                visited[qVW + wrd] |= bit;

                const float dist = tg_dist[g];
                float worst = 0.f;
                uint  widx  = beam_size;
                for (uint i = 0; i < beam_size; ++i) {
                    float cd   = cand_dists[qBS + i];
                    float absd = cd < 0.f ? (-cd - 1.f) : cd;
                    if (absd > worst) { worst = absd; widx = i; }
                }
                if (widx < beam_size && dist < worst) {
                    cand_dists[qBS + widx] = dist;
                    cand_nodes[qBS + widx] = nbr;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup_barrier(mem_flags::mem_device);
    }

    // ── Extract top-K (thread 0) ──────────────────────────────────────
    if (t_id == 0) {
        // Collect + sort valid candidates
        // Temporary sort in registers (K <= 32 typical)
        float  sd[512];
        uint   si[512];
        uint   n_valid = 0;
        for (uint i = 0; i < beam_size; ++i) {
            float cd = cand_dists[qBS + i];
            uint  cn = cand_nodes[qBS + i];
            if (cn == 0xFFFFFFFFu) continue;
            float absd = cd < 0.f ? (-cd - 1.f) : cd;
            sd[n_valid] = absd;
            si[n_valid] = cn;
            ++n_valid;
        }
        // Insertion sort — tie-break by node index ascending (FAISS contract)
        for (uint i = 1; i < n_valid; ++i) {
            float kd = sd[i]; uint kn = si[i];
            int j = (int)i - 1;
            while (j >= 0 && (sd[j] > kd || (sd[j] == kd && si[j] > kn))) {
                sd[j+1] = sd[j]; si[j+1] = si[j]; --j;
            }
            sd[j+1] = kd; si[j+1] = kn;
        }
        for (uint k = 0; k < K; ++k) {
            if (k < n_valid) {
                out_nbrs[qK + k]  = si[k];
                out_dists[qK + k] = sd[k];
            } else {
                out_nbrs[qK + k]  = 0xFFFFFFFFu;
                out_dists[qK + k] = INFINITY;
            }
        }
    }
}

// nn-descent: graph refinement kernel.
// Each threadgroup handles one node. Threads compute distances to 2-hop candidates.
// If any 2-hop neighbor improves the current k-NN list, the graph is updated.
// Returns 1 in `improved[node]` if any update was made (for convergence detection).

kernel void nn_descent(
    device const float* dataset    [[ buffer(0) ]],  // N×D
    device uint*        graph      [[ buffer(1) ]],  // N×G (mutable)
    device uint*        improved   [[ buffer(2) ]],  // N (output: 1=changed)
    device float*       graph_dist [[ buffer(3) ]],  // N×G current distances
    constant uint&      N          [[ buffer(4) ]],
    constant uint&      D          [[ buffer(5) ]],
    constant uint&      G          [[ buffer(6) ]],
    uint node_id                   [[ threadgroup_position_in_grid ]],
    uint t_id                      [[ thread_position_in_threadgroup ]],
    uint n_threads                 [[ threads_per_threadgroup ]])
{
    const ulong nBase = (ulong)node_id * G;
    const ulong dBase = (ulong)node_id * D;
    uint any_improved = 0;

    // Each thread processes a subset of this node's neighbors
    for (uint g = t_id; g < G; g += n_threads) {
        const uint nbr = graph[nBase + g];
        if (nbr >= N) continue;
        const ulong nbrBase = (ulong)nbr * G;

        // Examine 2-hop neighbors (neighbors of neighbor)
        for (uint h = 0; h < G; ++h) {
            const uint candidate = graph[nbrBase + h];
            if (candidate >= N || candidate == node_id) continue;

            // Check if candidate is already in node's graph
            bool already_present = false;
            for (uint i = 0; i < G; ++i) {
                if (graph[nBase + i] == candidate) { already_present = true; break; }
            }
            if (already_present) continue;

            // Compute L2 distance from node to candidate
            float dist = 0.f;
            for (uint d = 0; d < D; ++d) {
                float delta = dataset[dBase + d] - dataset[(ulong)candidate * D + d];
                dist += delta * delta;
            }

            // Find worst current neighbor
            float worst_d = 0.f;
            uint  worst_g = G;
            for (uint i = 0; i < G; ++i) {
                if (graph_dist[nBase + i] > worst_d) {
                    worst_d = graph_dist[nBase + i];
                    worst_g = i;
                }
            }
            if (worst_g < G && dist < worst_d) {
                graph[nBase + worst_g]      = candidate;
                graph_dist[nBase + worst_g] = dist;
                any_improved = 1;
            }
        }
    }
    if (any_improved) improved[node_id] = 1;
}

// P4: brute-force L2 distance — one thread per base vector.
// Each thread computes squared Euclidean distance from base vector gid to the query.
// dataset: N x D row-major float32
// query:   1 x D float32
// distances: N float32 output
kernel void l2_distance_kernel(
    device const float* dataset   [[ buffer(0) ]],
    device const float* query     [[ buffer(1) ]],
    device float*       distances [[ buffer(2) ]],
    constant uint&      D         [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]])
{
    float dist = 0.0f;
    const uint base = gid * D;
    for (uint d = 0; d < D; ++d) {
        float delta = dataset[base + d] - query[d];
        dist += delta * delta;
    }
    distances[gid] = dist;
}
