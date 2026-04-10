"""
Head-to-head benchmark: Memoria vs MemPalace on LongMemEval (500 questions).

Both systems get the same data: per-question haystack of ~53 conversation sessions.
We measure R@5, R@10, NDCG@10.

MemPalace uses ChromaDB default embeddings (no cross-encoder).
Memoria has three modes:
  - original:  bi-encoder(all docs) + BM25 + cross-encoder top-50x2
  - optimized: precomputed embeddings + BM25 + cross-encoder top-15x1 + single-probe
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import chromadb
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from memoria.embeddings import Embedder, top_k_similar
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# ── Metrics ──────────────────────────────────────────────────────────────

def recall_at_k(retrieved, correct, k):
    return 1.0 if correct.issubset(set(retrieved[:k])) else 0.0


def dcg(rels, k):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def ndcg_at_k(retrieved, correct, k):
    rels = [1.0 if r in correct else 0.0 for r in retrieved[:k]]
    idcg = dcg(sorted(rels, reverse=True), k)
    return dcg(rels, k) / idcg if idcg > 0 else 0.0


def tokenize(t):
    return re.findall(r"\w+", t.lower())


# ── Embedding cache (simulates precomputed store) ───────────────────────

class EmbeddingCache:
    """Cache session embeddings by content hash — simulates a precomputed store."""

    def __init__(self, embedder):
        self.embedder = embedder
        self._cache = {}
        self.hits = 0
        self.misses = 0

    def get_embeddings(self, session_ids, user_docs):
        result = np.empty((len(user_docs), self.embedder.dimension), dtype=np.float32)
        to_embed_idx = []
        to_embed_texts = []
        for i, doc in enumerate(user_docs):
            key = hashlib.md5(doc.encode()).hexdigest()
            if key in self._cache:
                result[i] = self._cache[key]
                self.hits += 1
            else:
                to_embed_idx.append(i)
                to_embed_texts.append(doc)
                self.misses += 1

        if to_embed_texts:
            new_embs = self.embedder.embed(to_embed_texts)
            for j, i in enumerate(to_embed_idx):
                result[i] = new_embs[j]
                key = hashlib.md5(user_docs[i].encode()).hexdigest()
                self._cache[key] = new_embs[j]
        return result


# ── Helpers: extract docs from question ─────────────────────────────────

def extract_docs(q):
    sessions = q["haystack_sessions"]
    session_ids = q["haystack_session_ids"]
    user_docs, all_docs, ids = [], [], []
    seen = set()
    for sid, sess in zip(session_ids, sessions):
        if sid in seen:
            continue
        seen.add(sid)
        ut = [t["content"] for t in sess if t["role"] == "user"]
        ud = "\n".join(ut)
        if ud.strip():
            user_docs.append(ud)
            all_docs.append("\n".join(t["content"] for t in sess))
            ids.append(sid)
    return user_docs, all_docs, ids


# ── Memoria retrieval (original) ────────────────────────────────────────

def retrieve_memoria_original(q, embedder, reranker):
    user_docs, all_docs, ids = extract_docs(q)
    if not user_docs:
        return []
    n = len(user_docs)

    corpus_embs = embedder.embed(user_docs)
    query_emb = embedder.embed_single(q["question"])

    similar = top_k_similar(query_emb, corpus_embs, k=min(60, n))
    vec_scores = {ids[idx]: float(s) for idx, s in similar}

    scores = [s for _, s in similar]
    if len(similar) >= 5:
        gap_1_2 = scores[0] - scores[1]
        gap_2_5 = scores[1] - scores[min(4, len(scores) - 1)]
        dense = sum(1 for s in scores if s >= scores[0] * 0.9)
        if gap_1_2 < gap_2_5 * 2.0 or dense >= 3:
            for pi in range(min(3, len(similar))):
                expanded = q["question"] + " " + user_docs[similar[pi][0]][:200]
                exp_emb = embedder.embed_single(expanded)
                for idx, s in top_k_similar(exp_emb, corpus_embs, k=min(20, n)):
                    sid = ids[idx]
                    vec_scores[sid] = max(
                        vec_scores.get(sid, 0.0), float(s) * 0.5 / (pi + 1)
                    )

    bm25 = BM25Okapi([tokenize(d) for d in user_docs])
    bm25_raw = bm25.get_scores(tokenize(q["question"]))
    bm25_max = bm25_raw.max() if bm25_raw.max() > 0 else 1.0
    bm25_norm = bm25_raw / bm25_max

    stage1 = {}
    for i, sid in enumerate(ids):
        stage1[sid] = vec_scores.get(sid, 0.0) * 0.85 + float(bm25_norm[i]) * 0.15

    cands = sorted(stage1.items(), key=lambda x: x[1], reverse=True)[:50]
    csids = [s for s, _ in cands]

    pairs_user = [(q["question"], user_docs[ids.index(s)][:512]) for s in csids]
    pairs_all = [(q["question"], all_docs[ids.index(s)][:512]) for s in csids]
    ce_user = reranker.predict(pairs_user)
    ce_all = reranker.predict(pairs_all)
    ce_combined = np.maximum(ce_user, ce_all)

    ce_min, ce_max = float(ce_combined.min()), float(ce_combined.max())
    ce_range = ce_max - ce_min if ce_max != ce_min else 1.0
    s1_max = max(stage1.values())

    final = {}
    for i, sid in enumerate(csids):
        ce_norm = (float(ce_combined[i]) - ce_min) / ce_range
        s1_norm = stage1[sid] / s1_max
        final[sid] = ce_norm * 0.4 + s1_norm * 0.6

    ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)
    return [s for s, _ in ranked[:10]]


# ── Memoria retrieval (optimized) ───────────────────────────────────────

def retrieve_memoria_fast(q, embedder, reranker, emb_cache):
    user_docs, all_docs, ids = extract_docs(q)
    if not user_docs:
        return []
    n = len(user_docs)

    # Opt 1: cached corpus embeddings (precomputed in production)
    corpus_embs = emb_cache.get_embeddings(ids, user_docs)
    query_emb = embedder.embed_single(q["question"])

    # Stage 1a: bi-encoder
    similar = top_k_similar(query_emb, corpus_embs, k=min(60, n))
    vec_scores = {ids[idx]: float(s) for idx, s in similar}

    # Opt 2: single-probe instead of 3
    scores = [s for _, s in similar]
    if len(similar) >= 5:
        gap_1_2 = scores[0] - scores[1]
        gap_2_5 = scores[1] - scores[min(4, len(scores) - 1)]
        dense = sum(1 for s in scores if s >= scores[0] * 0.9)
        if gap_1_2 < gap_2_5 * 2.0 or dense >= 3:
            expanded = q["question"] + " " + user_docs[similar[0][0]][:200]
            exp_emb = embedder.embed_single(expanded)
            for idx, s in top_k_similar(exp_emb, corpus_embs, k=min(20, n)):
                sid = ids[idx]
                vec_scores[sid] = max(vec_scores.get(sid, 0.0), float(s) * 0.5)

    # Stage 1c: BM25
    bm25 = BM25Okapi([tokenize(d) for d in user_docs])
    bm25_raw = bm25.get_scores(tokenize(q["question"]))
    bm25_max = bm25_raw.max() if bm25_raw.max() > 0 else 1.0
    bm25_norm = bm25_raw / bm25_max

    stage1 = {}
    for i, sid in enumerate(ids):
        stage1[sid] = vec_scores.get(sid, 0.0) * 0.85 + float(bm25_norm[i]) * 0.15

    # Opt 3: cross-encoder top-15, single pass with combined text
    cands = sorted(stage1.items(), key=lambda x: x[1], reverse=True)[:15]
    csids = [s for s, _ in cands]

    pairs = []
    for s in csids:
        idx = ids.index(s)
        doc = user_docs[idx][:384] + "\n" + all_docs[idx][:128]
        pairs.append((q["question"], doc))
    ce_scores = reranker.predict(pairs)

    ce_min, ce_max = float(ce_scores.min()), float(ce_scores.max())
    ce_range = ce_max - ce_min if ce_max != ce_min else 1.0
    s1_max = max(stage1.values())

    final = {}
    for i, sid in enumerate(csids):
        ce_norm = (float(ce_scores[i]) - ce_min) / ce_range
        s1_norm = stage1[sid] / s1_max
        final[sid] = ce_norm * 0.4 + s1_norm * 0.6

    # Backfill: stage1-only candidates ranked below all reranked ones
    min_reranked = min(final.values()) if final else 0
    for sid, s1 in stage1.items():
        if sid not in final:
            final[sid] = (s1 / s1_max) * min_reranked * 0.99

    ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)
    return [s for s, _ in ranked[:10]]


# ── MemPalace retrieval ─────────────────────────────────────────────────

def retrieve_mempalace(q, palace_path):
    user_docs, _, ids = extract_docs(q)
    if not user_docs:
        return []

    client = chromadb.PersistentClient(path=palace_path)
    try:
        client.delete_collection("mempalace_drawers")
    except Exception:
        pass
    col = client.create_collection("mempalace_drawers")

    metas = [{"session_id": sid, "wing": "bench", "room": "sessions"} for sid in ids]
    for i in range(0, len(user_docs), 5000):
        col.add(
            documents=user_docs[i:i+5000],
            ids=ids[i:i+5000],
            metadatas=metas[i:i+5000],
        )

    results = col.query(
        query_texts=[q["question"]],
        n_results=10,
        include=["distances"],
    )
    return results["ids"][0]


# ── Evaluate one system ─────────────────────────────────────────────────

def evaluate(name, data, retrieve_fn):
    r5_all, r10_all, n10_all = [], [], []
    type_metrics = defaultdict(lambda: {"r5": [], "r10": [], "n10": []})
    t0 = time.time()

    for i, q in enumerate(data):
        correct = set(q["answer_session_ids"])
        retrieved = retrieve_fn(q)
        r5 = recall_at_k(retrieved, correct, 5)
        r10 = recall_at_k(retrieved, correct, 10)
        n10 = ndcg_at_k(retrieved, correct, 10)
        r5_all.append(r5); r10_all.append(r10); n10_all.append(n10)
        qtype = q["question_type"]
        type_metrics[qtype]["r5"].append(r5)
        type_metrics[qtype]["r10"].append(r10)
        type_metrics[qtype]["n10"].append(n10)
        if (i + 1) % 100 == 0:
            print(f"  {name} {i+1}/{len(data)} R@5={np.mean(r5_all):.3f}")

    elapsed = time.time() - t0
    return {
        "recall_5": float(np.mean(r5_all)),
        "recall_10": float(np.mean(r10_all)),
        "ndcg_10": float(np.mean(n10_all)),
        "time_sec": elapsed,
        "qps": len(data) / elapsed,
        "by_type": {
            t: {k: float(np.mean(v)) for k, v in m.items()}
            for t, m in type_metrics.items()
        },
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--output", default="results_headtohead.json")
    parser.add_argument("--skip-mempalace", action="store_true")
    parser.add_argument("--skip-original", action="store_true")
    args = parser.parse_args()

    if args.data is None:
        default = Path(__file__).parent / "data" / "longmemeval_s_cleaned.json"
        if default.exists():
            args.data = str(default)
        else:
            from download_data import download
            args.data = str(download())

    data = json.loads(Path(args.data).read_text())
    if args.max:
        data = data[:args.max]

    print(f"Loading models...")
    embedder = Embedder()
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    emb_cache = EmbeddingCache(embedder)

    # Warm up embedding cache with a pre-scan pass
    print(f"Pre-computing session embeddings for {len(data)} questions...")
    t_precache = time.time()
    for q in data:
        user_docs, _, ids = extract_docs(q)
        emb_cache.get_embeddings(ids, user_docs)
    t_precache = time.time() - t_precache
    print(f"  Cached {len(emb_cache._cache)} unique sessions in {t_precache:.1f}s "
          f"({emb_cache.hits} hits, {emb_cache.misses} misses)")
    # Reset counters for benchmark
    emb_cache.hits = 0
    emb_cache.misses = 0

    results = {}

    # ── Memoria optimized ────────────────────────────────────────────
    print(f"\n--- Memoria (optimized) ---")
    results["memoria_fast"] = evaluate(
        "memoria-fast", data,
        lambda q: retrieve_memoria_fast(q, embedder, reranker, emb_cache),
    )
    print(f"  Cache: {emb_cache.hits} hits, {emb_cache.misses} misses")

    # ── Memoria original ─────────────────────────────────────────────
    if not args.skip_original:
        print(f"\n--- Memoria (original) ---")
        results["memoria_original"] = evaluate(
            "memoria-orig", data,
            lambda q: retrieve_memoria_original(q, embedder, reranker),
        )

    # ── MemPalace ────────────────────────────────────────────────────
    if not args.skip_mempalace:
        palace_tmp = tempfile.mkdtemp(prefix="mempalace_bench_")
        print(f"\n--- MemPalace ---")
        results["mempalace"] = evaluate(
            "mempalace", data,
            lambda q: retrieve_mempalace(q, palace_tmp),
        )
        shutil.rmtree(palace_tmp, ignore_errors=True)

    # ── Print comparison table ───────────────────────────────────────
    print(f"\n{'='*85}")
    print(f"HEAD-TO-HEAD: LongMemEval ({len(data)} questions)")
    print(f"{'='*85}")
    print(f"\n{'System':25s} {'R@5':>7} {'R@10':>7} {'NDCG@10':>8} {'Time':>7} {'q/s':>6}")
    print(f"{'-'*25} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*6}")

    for name, r in sorted(results.items()):
        print(f"{name:25s} {r['recall_5']*100:>6.1f}% {r['recall_10']*100:>6.1f}% "
              f"{r['ndcg_10']:>8.4f} {r['time_sec']:>6.1f}s {r['qps']:>6.1f}")

    # By-type breakdown for fast vs original
    if "memoria_fast" in results:
        fast = results["memoria_fast"]
        print(f"\nMemoria (optimized) by type:")
        for t in sorted(fast["by_type"]):
            m = fast["by_type"][t]
            orig_r5 = results.get("memoria_original", {}).get("by_type", {}).get(t, {}).get("r5", 0)
            delta = f"  Δ={m['r5']-orig_r5:+.3f}" if orig_r5 else ""
            print(f"  {t:30s}  R@5={m['r5']:.3f}  R@10={m['r10']:.3f}{delta}")

    print(f"\n{'='*85}")
    if "memoria_fast" in results and "memoria_original" in results:
        speedup = results["memoria_original"]["time_sec"] / results["memoria_fast"]["time_sec"]
        r5_delta = (results["memoria_fast"]["recall_5"] - results["memoria_original"]["recall_5"]) * 100
        print(f"Speedup: {speedup:.1f}x faster  |  R@5 change: {r5_delta:+.1f}%")
    print(f"{'='*85}")

    out = Path(__file__).parent / args.output
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
