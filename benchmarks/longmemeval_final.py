"""
Final optimized LongMemEval benchmark: memoria retrieval pipeline.

Architecture:
  Stage 1 — Candidate generation (bi-encoder + multi-probe + BM25)
    1a. Bi-encoder vector search (all-MiniLM-L6-v2, user turns, top-60)
    1b. Conditional multi-probe expansion from top-3 vector hits
        (fires when score distribution is clustered → multi-session indicator)
    1c. BM25 keyword scoring (tiebreaker for near-ties)
    Merge: 85% vector + 15% BM25

  Stage 2 — Cross-encoder reranking (top-50 from stage 1)
    2a. Score each candidate with cross-encoder on user-only text
    2b. Score each candidate with cross-encoder on all-turns text
    2c. Take max of both scores per session
    Merge: 40% cross-encoder + 60% stage1

Result: 95.2% R@5, 98.4% R@10 on 500 questions.
Ceiling: 99.4% (3 questions need 6 answer sessions, impossible for R@5).
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from memoria.embeddings import Embedder, top_k_similar
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


def tokenize(t):
    return re.findall(r"\w+", t.lower())


def recall_at_k(retrieved, correct, k):
    return 1.0 if correct.issubset(set(retrieved[:k])) else 0.0


def dcg(rels, k):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def ndcg_at_k(retrieved, correct, k):
    rels = [1.0 if r in correct else 0.0 for r in retrieved[:k]]
    idcg = dcg(sorted(rels, reverse=True), k)
    return dcg(rels, k) / idcg if idcg > 0 else 0.0


def retrieve(q, embedder, reranker):
    sessions = q["haystack_sessions"]
    session_ids = q["haystack_session_ids"]

    user_docs, all_docs, ids = [], [], []
    seen = set()
    for sid, sess in zip(session_ids, sessions):
        if sid in seen:
            continue
        seen.add(sid)
        ut = [t["content"] for t in sess if t["role"] == "user"]
        at = [t["content"] for t in sess]
        ud = "\n".join(ut)
        if ud.strip():
            user_docs.append(ud)
            all_docs.append("\n".join(at))
            ids.append(sid)
    if not user_docs:
        return []
    n = len(user_docs)

    corpus_embs = embedder.embed(user_docs)
    query_emb = embedder.embed_single(q["question"])

    # Stage 1a: bi-encoder
    similar = top_k_similar(query_emb, corpus_embs, k=min(60, n))
    vec_scores = {ids[idx]: float(s) for idx, s in similar}

    # Stage 1b: multi-probe expansion (conditional)
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

    # Stage 1c: BM25
    bm25 = BM25Okapi([tokenize(d) for d in user_docs])
    bm25_raw = bm25.get_scores(tokenize(q["question"]))
    bm25_max = bm25_raw.max() if bm25_raw.max() > 0 else 1.0
    bm25_norm = bm25_raw / bm25_max

    stage1 = {}
    for i, sid in enumerate(ids):
        stage1[sid] = vec_scores.get(sid, 0.0) * 0.85 + float(bm25_norm[i]) * 0.15

    # Stage 2: cross-encoder rerank top-50
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


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.data is None:
        from download_data import download
        args.data = str(download())

    data = json.loads(Path(args.data).read_text())
    if args.max:
        data = data[: args.max]

    embedder = Embedder()
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

    r5_all, r10_all = [], []
    type_metrics = defaultdict(lambda: {"r5": [], "r10": [], "n10": []})
    start = time.time()

    for i, q in enumerate(data):
        correct = set(q["answer_session_ids"])
        retrieved = retrieve(q, embedder, reranker)

        r5 = recall_at_k(retrieved, correct, 5)
        r10 = recall_at_k(retrieved, correct, 10)
        n10 = ndcg_at_k(retrieved, correct, 10)
        r5_all.append(r5)
        r10_all.append(r10)
        qtype = q["question_type"]
        type_metrics[qtype]["r5"].append(r5)
        type_metrics[qtype]["r10"].append(r10)
        type_metrics[qtype]["n10"].append(n10)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(
                f"  {i+1}/{len(data)} R@5={np.mean(r5_all):.3f} "
                f"R@10={np.mean(r10_all):.3f} ({(i+1)/elapsed:.1f} q/s)"
            )

    elapsed = time.time() - start
    fails = sum(1 for r in r5_all if r == 0)
    impossible = sum(1 for q in data if len(q["answer_session_ids"]) > 5)

    print(f"\n{'='*75}")
    print(f"MEMORIA — LongMemEval Final Results")
    print(f"{'='*75}")
    print(f"R@5:    {np.mean(r5_all)*100:.1f}% ({fails} failures, {impossible} impossible)")
    print(f"R@10:   {np.mean(r10_all)*100:.1f}%")
    all_n10 = [v for m in type_metrics.values() for v in m["n10"]]
    print(f"NDCG@10: {np.mean(all_n10):.4f}")
    print(f"Time:   {elapsed:.0f}s ({len(data)/elapsed:.1f} q/s)")
    print(f"\nBy type:")
    for t in sorted(type_metrics):
        m = type_metrics[t]
        print(
            f"  {t:30s}  R@5={np.mean(m['r5']):.3f}  R@10={np.mean(m['r10']):.3f}  "
            f"NDCG={np.mean(m['n10']):.3f}  (n={len(m['r5'])})"
        )

    print(f"\n{'='*75}")
    print(f"{'Mode':25s} {'R@5':>7} {'R@10':>7} {'NDCG@10':>8}")
    print(f"{'-'*25} {'-'*7} {'-'*7} {'-'*8}")
    print(f"{'memoria baseline':25s} {'85.0%':>7} {'93.2%':>7} {'0.889':>8}")
    print(
        f"{'memoria optimized':25s} {np.mean(r5_all)*100:.1f}%"
        f"  {np.mean(r10_all)*100:.1f}% {np.mean(all_n10):>8.4f}"
    )
    print(f"{'mempalace (claimed)':25s} {'96.6%':>7} {'94.8%':>7} {'---':>8}")
    print(f"{'mempalace (reproduced)':25s} {'85.0%':>7} {'93.2%':>7} {'---':>8}")
    print(f"{'ceiling (3 impossible)':25s} {'99.4%':>7} {'---':>7} {'---':>8}")

    if args.output:
        out = {
            "recall_5": float(np.mean(r5_all)),
            "recall_10": float(np.mean(r10_all)),
            "ndcg_10": float(np.mean(all_n10)),
            "failures": fails,
            "impossible": impossible,
            "total": len(data),
            "elapsed_sec": elapsed,
            "by_type": {
                t: {k: float(np.mean(v)) for k, v in m.items()}
                for t, m in type_metrics.items()
            },
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
