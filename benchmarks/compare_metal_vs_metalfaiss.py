#!/usr/bin/env python3
"""
Compare cuvs-silicon (Metal GPU brute-force) vs metalfaiss (MLX-based FAISS)
on Cohere Wikipedia 1024d embeddings.

Usage:
  uv run --with mlx --with numpy \
    --with "metalfaiss @ file:///tmp/faiss_mlx_src/python" \
    python3 benchmarks/compare_metal_vs_metalfaiss.py \
    data/10k/cohere_wiki_base.bin \
    data/10k/cohere_wiki_queries.bin \
    data/10k/cohere_wiki_gt.bin
"""
import argparse
import json
import os
import struct
import subprocess
import time

import numpy as np

METAL_BINARY = os.path.join(
    os.path.dirname(__file__), "..", "build", "cohere_wiki_validation"
)


# ── Data loading ──────────────────────────────────────────────────────────

def load_bin_float32(path):
    with open(path, "rb") as f:
        rows, cols = struct.unpack("ii", f.read(8))
        data = np.frombuffer(f.read(rows * cols * 4), dtype=np.float32)
    return data.reshape(rows, cols)


def load_bin_int32(path):
    with open(path, "rb") as f:
        rows, cols = struct.unpack("ii", f.read(8))
        data = np.frombuffer(f.read(rows * cols * 4), dtype=np.int32)
    return data.reshape(rows, cols)


# ── Recall ────────────────────────────────────────────────────────────────

def recall_at_k(pred_indices, gt_indices, k):
    hits = 0
    Q = len(pred_indices)
    for q in range(Q):
        pred_set = set(pred_indices[q][:k].tolist())
        gt_set   = set(gt_indices[q][:k].tolist())
        hits += len(pred_set & gt_set)
    return hits / (Q * k)


# ── metalfaiss (MLX) benchmark ───────────────────────────────────────────

def run_metalfaiss(base, queries, gt, k, warmup=3, measure=10):
    try:
        from metalfaiss.indexflat import FlatIndex
    except ImportError as e:
        print(f"  metalfaiss not available: {e}")
        return None

    N, D = base.shape
    Q    = len(queries)

    index = FlatIndex(D)
    t_build_start = time.perf_counter()
    index.add(base)
    t_build = time.perf_counter() - t_build_start

    import mlx.core as mx

    # Warmup — force eval to ensure MLX JIT compilation before timing
    for _ in range(warmup):
        r = index.search(queries, k)
        mx.eval(r.distances, r.indices)  # force lazy evaluation

    # Measure — force eval inside timing to capture actual GPU execution
    latencies = []
    last_result = None
    for _ in range(measure):
        t0 = time.perf_counter()
        r = index.search(queries, k)
        mx.eval(r.distances, r.indices)  # block until GPU finishes
        latencies.append((time.perf_counter() - t0) * 1000)
        last_result = r

    pred = np.array(last_result.indices, dtype=np.int32)
    rec  = recall_at_k(pred, gt, k)
    latencies.sort()
    total_ms = sum(latencies)
    qps  = Q * measure / (total_ms / 1000)
    p50  = latencies[int(measure * 0.50)]
    p99  = latencies[max(0, int(measure * 0.99) - 1)]

    return dict(recall=rec, qps=qps, p50_ms=p50, p99_ms=p99,
                build_s=t_build, backend="metalfaiss/MLX")


# ── cuvs-silicon C++ benchmark ─────────────────────────────────────────────

def run_metal_cagra(base_path, queries_path, gt_path, k, warmup=3, measure=10):
    binary = os.path.abspath(METAL_BINARY)
    if not os.path.exists(binary):
        print(f"  binary not found: {binary}")
        return None

    cmd = [binary, base_path, queries_path, gt_path,
           str(k), str(warmup), str(measure)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode not in (0, 2):
        print("  cuvs-silicon failed:\n", result.stderr[:300])
        return None

    out = result.stdout

    def parse(key):
        for line in out.splitlines():
            if key in line:
                try:
                    return float(line.split("=")[-1].strip())
                except ValueError:
                    pass
        return None

    return dict(
        recall  = parse("recall@"),
        qps     = parse("QPS"),
        p50_ms  = parse("p50 ms"),
        p99_ms  = parse("p99 ms"),
        build_s = None,
        backend = "cuvs-silicon/Metal",
    )


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("queries")
    ap.add_argument("gt")
    ap.add_argument("--k",       type=int, default=10)
    ap.add_argument("--warmup",  type=int, default=3)
    ap.add_argument("--measure", type=int, default=10)
    args = ap.parse_args()

    base    = load_bin_float32(args.base)
    queries = load_bin_float32(args.queries)
    gt      = load_bin_int32(args.gt)
    N, D = base.shape
    Q    = len(queries)

    print(f"\nCohere Wikipedia Comparison: metalfaiss vs cuvs-silicon")
    print(f"  N={N:,}  D={D}  Q={Q}  K={args.k}")
    print(f"  warmup={args.warmup}  measure={args.measure}\n")

    print("[1] metalfaiss (MLX/Metal)")
    mf = run_metalfaiss(base, queries, gt, args.k, args.warmup, args.measure)
    if mf:
        print(f"  recall@{args.k} = {mf['recall']:.4f}")
        print(f"  QPS        = {mf['qps']:.1f}")
        print(f"  p50 ms     = {mf['p50_ms']:.2f}")
        print(f"  p99 ms     = {mf['p99_ms']:.2f}")
        print(f"  build time = {mf['build_s']:.3f}s")

    print(f"\n[2] cuvs-silicon (Metal Compute Shader)")
    mc = run_metal_cagra(args.base, args.queries, args.gt,
                          args.k, args.warmup, args.measure)
    if mc:
        print(f"  recall@{args.k} = {mc['recall']:.4f}")
        print(f"  QPS        = {mc['qps']:.1f}")
        print(f"  p50 ms     = {mc['p50_ms']:.2f}")
        print(f"  p99 ms     = {mc['p99_ms']:.2f}")

    if mf and mc:
        print(f"\n[Comparison]")
        qps_ratio = mc['qps'] / mf['qps'] if mf['qps'] else float('inf')
        p99_ratio = mc['p99_ms'] / mf['p99_ms'] if mf['p99_ms'] else float('inf')
        print(f"  cuvs-silicon QPS / metalfaiss QPS = {qps_ratio:.2f}x")
        print(f"  cuvs-silicon p99 / metalfaiss p99 = {p99_ratio:.2f}x")

        wins = []
        if mc['recall'] is not None and mf['recall'] is not None:
            if mc['recall'] >= mf['recall']:
                wins.append(f"recall@{args.k} [OK]")
        if mc['qps'] is not None and mf['qps'] is not None:
            if mc['qps'] > mf['qps']:
                wins.append("QPS [OK]")
        if mc['p99_ms'] is not None and mf['p99_ms'] is not None:
            if mc['p99_ms'] < mf['p99_ms']:
                wins.append("p99 [OK]")

        print(f"  cuvs-silicon wins: {', '.join(wins) if wins else 'none'}")

        os.makedirs("benchmarks", exist_ok=True)
        report = {"metal_cagra": mc, "metalfaiss": mf,
                  "dataset": {"N": N, "D": D, "Q": Q, "K": args.k}}
        out_path = "benchmarks/comparison_result.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  result saved: {out_path}")


if __name__ == "__main__":
    main()
