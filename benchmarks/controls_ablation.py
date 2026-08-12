"""Decompose memoria-balanced's edge over hybrid-ce-dual (90.5 -> 94.3 R-All@5).

Remaining components beyond hybrid candidates + dual-pass CE:
  EXP    stage-1b conditional single-probe query expansion
  BLEND  final score = 0.40*CE_minmax + 0.60*stage1_norm (vs pure CE ordering)

Arms (all: memoria stage1 fusion, K=15 candidates, dual-pass CE):
  blend-only   no expansion, memoria blend          = memoria minus EXP
  exp-only     expansion, pure CE ordering          = memoria minus BLEND
  exp+blend    expansion + blend                    = replication of memoria balanced
               (sanity: must reproduce 94.3/98.1/97.1 digit-for-digit)
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from memoria.embeddings import Embedder, top_k_similar
from memoria.retriever import EmbeddingCache, _tokenize
from longmemeval_final import (
    build_session_documents,
    is_official_retrieval_instance,
    user_evidence_session_ids,
)
from retrieval_metrics import metric_bundle

K_CE = 15
ARMS = ["blend-only", "exp-only", "exp+blend"]


def stage1_scores(ids, vec_scores, bnorm):
    dense_max = max((max(s, 0.0) for s in vec_scores.values()), default=0.0)
    scale = dense_max if dense_max > 0 else 1.0
    return {
        sid: max(vec_scores.get(sid, 0.0), 0.0) / scale * 0.70 + float(bnorm[i]) * 0.30
        for i, sid in enumerate(ids)
    }


def dual_ce(reranker, query, cand_ids, idx, user_docs, all_docs):
    pp = [(query, user_docs[idx[c]][:512]) for c in cand_ids]
    sp = [(query, all_docs[idx[c]][:512] or user_docs[idx[c]][:512]) for c in cand_ids]
    return np.maximum(
        np.asarray(reranker.predict(pp), dtype=float),
        np.asarray(reranker.predict(sp), dtype=float),
    )


def rank_pure_ce(cand_ids, cross, ids, stage1):
    order = np.argsort(cross)[::-1]
    ranked = [cand_ids[int(i)] for i in order]
    rest = sorted((s for s in ids if s not in set(cand_ids)),
                  key=lambda s: stage1[s], reverse=True)
    return ranked + rest


def rank_blend(cand_ids, cross, ids, stage1):
    """Replicate _hybrid_rank's final fusion exactly."""
    cross = cross.reshape(-1)
    cmin = float(cross.min()) if cross.size else 0.0
    cmax = float(cross.max()) if cross.size else 0.0
    crange = cmax - cmin
    s1max = max(stage1.values(), default=0.0)
    s1scale = s1max if s1max > 0 else 1.0
    final = {}
    for i, cid in enumerate(cand_ids):
        cn = (float(cross[i]) - cmin) / crange if crange > 0 else 0.5
        final[cid] = cn * 0.40 + stage1[cid] / s1scale * 0.60
    mn = min(final.values(), default=0.0)
    for sid, sc in stage1.items():
        if sid not in final:
            final[sid] = sc / s1scale * mn * 0.99
    return [s for s, _ in sorted(final.items(), key=lambda kv: kv[1], reverse=True)]


def main():
    data_path = Path(__file__).parent / "data" / "longmemeval_s_cleaned.json"
    source = json.loads(data_path.read_text())
    data = [q for q in source if is_official_retrieval_instance(q)]
    print(f"scored rows: {len(data)}")

    embedder = Embedder()
    cache = EmbeddingCache(embedder)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

    metric_names = ["recall_any@5", "recall_all@5", "ndcg_any@5",
                    "recall_any@10", "recall_all@10", "ndcg_any@10"]
    agg = {a: {m: [] for m in metric_names} for a in ARMS}
    by_type = {a: defaultdict(lambda: {m: [] for m in metric_names}) for a in ARMS}
    start = time.time()

    for qi, q in enumerate(data):
        correct = user_evidence_session_ids(q)
        ids, user_docs, all_docs = build_session_documents(q)
        n = len(ids)
        idx = {sid: i for i, sid in enumerate(ids)}
        query = q["question"]
        qtype = q["question_type"]

        corpus = cache.get_batch(user_docs)
        q_emb = embedder.embed_single(query)

        # stage 1a: dense pool exactly as retrieve_sessions
        similar = top_k_similar(q_emb, corpus, k=min(60, n))
        vec_base = {ids[i]: float(s) for i, s in similar}

        # stage 1b: conditional expansion exactly as retrieve_sessions
        vec_exp = dict(vec_base)
        scores = [s for _, s in similar]
        if len(similar) >= 5:
            gap_1_2 = scores[0] - scores[1]
            gap_2_5 = scores[1] - scores[min(4, len(scores) - 1)]
            dense_ct = sum(1 for s in scores if s >= scores[0] * 0.9)
            if gap_1_2 < gap_2_5 * 2.0 or dense_ct >= 3:
                expanded = query + " " + user_docs[similar[0][0]][:200]
                exp_emb = embedder.embed_single(expanded)
                for i, s in top_k_similar(exp_emb, corpus, k=min(20, n)):
                    sid = ids[i]
                    vec_exp[sid] = max(vec_exp.get(sid, 0.0), float(s) * 0.5)

        # shared BM25 (memoria tokenizer)
        bm25 = BM25Okapi([_tokenize(d) for d in user_docs])
        braw = np.asarray(bm25.get_scores(_tokenize(query)), dtype=float)
        bmax = float(braw.max()) if braw.size else 0.0
        bnorm = braw / bmax if bmax > 0 else np.zeros_like(braw)

        s1_noexp = stage1_scores(ids, vec_base, bnorm)
        s1_exp = stage1_scores(ids, vec_exp, bnorm)

        cand_noexp = [s for s, _ in sorted(s1_noexp.items(), key=lambda kv: kv[1],
                                           reverse=True)[:min(K_CE, n)]]
        cand_exp = [s for s, _ in sorted(s1_exp.items(), key=lambda kv: kv[1],
                                         reverse=True)[:min(K_CE, n)]]

        cross_noexp = dual_ce(reranker, query, cand_noexp, idx, user_docs, all_docs)
        cross_exp = (cross_noexp if cand_exp == cand_noexp
                     else dual_ce(reranker, query, cand_exp, idx, user_docs, all_docs))

        retrieved = {
            "blend-only": rank_blend(cand_noexp, cross_noexp, ids, s1_noexp),
            "exp-only": rank_pure_ce(cand_exp, cross_exp, ids, s1_exp),
            "exp+blend": rank_blend(cand_exp, cross_exp, ids, s1_exp),
        }

        for a in ARMS:
            m = metric_bundle(retrieved[a], correct, 5)
            m.update(metric_bundle(retrieved[a], correct, 10))
            for name, value in m.items():
                agg[a][name].append(value)
                by_type[a][qtype][name].append(value)

        if (qi + 1) % 100 == 0:
            el = time.time() - start
            print(f"  {qi+1}/{len(data)} ({(qi+1)/el:.2f} q/s)  " + "  ".join(
                f"{a}:R-all@5={np.mean(agg[a]['recall_all@5']):.3f}" for a in ARMS))

    el = time.time() - start
    out = {"k_ce": K_CE, "rows": len(data), "elapsed_sec": el, "arms": {}}
    print(f"\n{'arm':12s} {'R-any@5':>8s} {'R-all@5':>8s} {'NDCG@5':>8s} "
          f"{'R-any@10':>9s} {'R-all@10':>9s} {'NDCG@10':>8s}")
    for a in ARMS:
        m = {name: float(np.mean(v)) for name, v in agg[a].items()}
        out["arms"][a] = {
            "metrics": m,
            "by_type": {t: {name: float(np.mean(v)) for name, v in tm.items()}
                        for t, tm in by_type[a].items()},
        }
        print(f"{a:12s} {m['recall_any@5']*100:7.1f}% {m['recall_all@5']*100:7.1f}% "
              f"{m['ndcg_any@5']:8.4f} {m['recall_any@10']*100:8.1f}% "
              f"{m['recall_all@10']*100:8.1f}% {m['ndcg_any@10']:8.4f}")
    print("reference: hybrid-ce-dual 96.9/90.5/0.9137/98.6/95.9/0.9250 ; "
          "memoria audited 98.1/94.3/0.9414/99.0/97.1/0.9478")

    print("\nby type — exp+blend (memoria replication):")
    for t in sorted(out["arms"]["exp+blend"]["by_type"]):
        tm = out["arms"]["exp+blend"]["by_type"][t]
        print(f"  {t:30s} R-all@5={tm['recall_all@5']:.3f} "
              f"R-all@10={tm['recall_all@10']:.3f} NDCG@10={tm['ndcg_any@10']:.3f}")

    out_path = Path(__file__).parent / "results_controls_ablation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved {out_path} ({el:.0f}s)")


if __name__ == "__main__":
    main()
