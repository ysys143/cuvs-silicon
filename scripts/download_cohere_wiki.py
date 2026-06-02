#!/usr/bin/env python3
"""Download a subset of CohereLabs Wikipedia embeddings for Metal CAGRA validation.

Saves three binary files in flat float32 format:
  cohere_wiki_base.bin    -- N_BASE x DIM float32 base vectors
  cohere_wiki_queries.bin -- N_QUERY x DIM float32 query vectors
  cohere_wiki_gt.bin      -- N_QUERY x K  int32   ground-truth indices (brute-force)

Usage:
  python3 scripts/download_cohere_wiki.py --base 10000 --queries 100 --k 10 --out data/
"""
import argparse
import os
import struct
import sys

import numpy as np
from datasets import load_dataset

N_BASE_DEFAULT    = 10_000
N_QUERY_DEFAULT   = 100
K_DEFAULT         = 10
DIM               = 1024   # Cohere Embed Multilingual v3 output dimension
DATASET_NAME      = "CohereLabs/wikipedia-2023-11-embed-multilingual-v3"
DATASET_CONFIG    = "en"


def load_embeddings(n: int, skip: int = 0) -> np.ndarray:
    """Stream n embeddings from the English Wikipedia dataset."""
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG,
                      split="train", streaming=True)
    embs = []
    for i, row in enumerate(ds):
        if i < skip:
            continue
        emb = np.array(row["emb"], dtype=np.float32)
        if emb.shape[0] != DIM:
            continue
        embs.append(emb)
        if len(embs) >= n:
            break
        if len(embs) % 1000 == 0:
            print(f"  loaded {len(embs)}/{n}", flush=True)
    return np.stack(embs)


def brute_force_gt(base: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Compute ground-truth top-k indices using L2 distance (CPU, exact)."""
    gt = np.empty((len(queries), k), dtype=np.int32)
    for i, q in enumerate(queries):
        diffs = base - q           # N x DIM
        dists = (diffs ** 2).sum(axis=1)
        gt[i] = np.argsort(dists)[:k]
    return gt


def save_float32_bin(arr: np.ndarray, path: str) -> None:
    arr = arr.astype(np.float32)
    with open(path, "wb") as f:
        # Header: rows (int32), cols (int32)
        f.write(struct.pack("ii", arr.shape[0], arr.shape[1]))
        arr.tofile(f)
    print(f"  saved {path}  shape={arr.shape}")


def save_int32_bin(arr: np.ndarray, path: str) -> None:
    arr = arr.astype(np.int32)
    with open(path, "wb") as f:
        f.write(struct.pack("ii", arr.shape[0], arr.shape[1]))
        arr.tofile(f)
    print(f"  saved {path}  shape={arr.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",    type=int, default=N_BASE_DEFAULT)
    parser.add_argument("--queries", type=int, default=N_QUERY_DEFAULT)
    parser.add_argument("--k",       type=int, default=K_DEFAULT)
    parser.add_argument("--out",     type=str, default="data")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Downloading {args.base} base vectors from {DATASET_NAME} ({DATASET_CONFIG})...")
    base = load_embeddings(args.base)
    print(f"Downloading {args.queries} query vectors (offset {args.base})...")
    queries = load_embeddings(args.queries, skip=args.base)

    print(f"Computing brute-force ground truth (k={args.k})...")
    gt = brute_force_gt(base, queries, args.k)

    save_float32_bin(base,    os.path.join(args.out, "cohere_wiki_base.bin"))
    save_float32_bin(queries, os.path.join(args.out, "cohere_wiki_queries.bin"))
    save_int32_bin  (gt,      os.path.join(args.out, "cohere_wiki_gt.bin"))

    print("Done.")


if __name__ == "__main__":
    main()
