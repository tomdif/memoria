"""Attribution controls: commodity cross-encoder rerank baselines for LongMemEval-S.

Each control uses the SAME cross-encoder (ms-marco-MiniLM-L-12-v2), the SAME
candidate depth as memoria balanced (K=15), the SAME official scope, session
documents, and metric implementation as longmemeval_final.py.

Controls:
  bm25-ce        BM25 (official whitespace tokenization) top-15 -> CE(user_doc[:512]) ordering
  minilm-ce      MiniLM dense top-15 -> CE(user_doc[:512]) ordering
  hybrid-ce      memoria stage1 fusion (0.70*dense + 0.30*bm25, memoria tokenizer,
                 dense from top-60 like memoria, NO query expansion) top-15
                 -> CE(user_doc[:512]) ordering  [isolates hybrid candidates]
  hybrid-ce-dual same candidates as hybrid-ce, CE = max(user[:512], all[:512])
                 [adds memoria's dual-pass rendering, still pure CE ordering,
                  no 0.4/0.6 score blending, no expansion]
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

from memoria.embeddings import Embedder
from memoria.retriever import EmbeddingCache, _tokenize
from longmemeval_final import (
    build_session_documents,
    is_official_retrieval_instance,
    user_evidence_session_ids,
)
from retrieval_metrics import metric_bundle

K_CE = 15  # matches _RERANK_CONFIG[RetrievalMode.BALANCED]
DENSE_POOL = 60  # matches memoria retrieve_sessions top_k_similar(k=min(60, n))

CONTROLS = ["bm25-ce", "minilm-ce", "hybrid-ce", "hybrid-ce-dual"]


def ce_order(reranker, query, candidate_ids, id_to_index, user_docs, all_docs, dual):
    primary_pairs = [(query, user_docs[id_to_index[cid]][:512]) for cid in candidate_ids]
    scores = np.asarray(reranker.predict(primary_pairs), dtype=float)
    if dual:
        source_pairs = [
            (query, all_docs[id_to_index[cid]][:512] or user_docs[id_to_index[cid]][:512])
            for cid in candidate_ids
        ]
        scores = np.maximum(scores, np.asarray(reranker.predict(source_pairs), dtype=float))
    order = np.argsort(scores)[::-1]
    return [candidate_ids[int(i)] for i in order]


def main():
    data_path = Path(__file__).parent / "data" / "longmemeval_s_cleaned.json"
    source = json.loads(data_path.read_text())
    data = [q for q in source if is_official_retrieval_instance(q)]
    print(f"scored rows: {len(data)} / {len(source)}")

    embedder = Embedder()
    cache = EmbeddingCache(embedder)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

    metric_names = [
        "recall_any@5", "recall_all@5", "ndcg_any@5",
        "recall_any@10", "recall_all@10", "ndcg_any@10",
    ]
    agg = {c: {m: [] for m in metric_names} for c in CONTROLS}
    by_type = {c: defaultdict(lambda: {m: [] for m in metric_names}) for c in CONTROLS}
    start = time.time()

    for qi, q in enumerate(data):
        correct = user_evidence_session_ids(q)
        ids, user_docs, all_docs = build_session_documents(q)
        n = len(ids)
        id_to_index = {sid: i for i, sid in enumerate(ids)}
        query = q["question"]
        qtype = q["question_type"]

        # Dense scores (shared)
        corpus = cache.get_batch(user_docs)
        q_emb = embedder.embed_single(query)
        dense_all = np.asarray(corpus @ q_emb, dtype=float)

        # -------- bm25-ce: official whitespace BM25 top-K -> CE
        bm25_official = BM25Okapi([d.split(" ") for d in user_docs])
        s = np.asarray(bm25_official.get_scores(query.split(" ")), dtype=float)
        bm25_top = [ids[int(i)] for i in np.argsort(s)[::-1][:K_CE]]
        rest = [sid for i, sid in enumerate(ids) if sid not in set(bm25_top)]
        retrieved = {"bm25-ce": ce_order(reranker, query, bm25_top, id_to_index,
                                         user_docs, all_docs, dual=False) + rest}

        # -------- minilm-ce: dense top-K -> CE
        dense_top = [ids[int(i)] for i in np.argsort(dense_all)[::-1][:K_CE]]
        rest = [sid for sid in ids if sid not in set(dense_top)]
        retrieved["minilm-ce"] = ce_order(reranker, query, dense_top, id_to_index,
                                          user_docs, all_docs, dual=False) + rest

        # -------- hybrid stage1 (memoria fusion, no expansion) top-K
        # dense pool of 60 like memoria's retrieve_sessions, clamped at 0
        pool = np.argsort(dense_all)[::-1][: min(DENSE_POOL, n)]
        dense_scores = {ids[int(i)]: max(float(dense_all[int(i)]), 0.0) for i in pool}
        dense_max = max(dense_scores.values(), default=0.0)
        dense_scale = dense_max if dense_max > 0 else 1.0
        bm25_mem = BM25Okapi([_tokenize(d) for d in user_docs])
        braw = np.asarray(bm25_mem.get_scores(_tokenize(query)), dtype=float)
        bmax = float(braw.max()) if braw.size else 0.0
        bnorm = braw / bmax if bmax > 0 else np.zeros_like(braw)
        stage1 = {
            sid: dense_scores.get(sid, 0.0) / dense_scale * 0.70 + float(bnorm[i]) * 0.30
            for i, sid in enumerate(ids)
        }
        hybrid_top = [sid for sid, _ in
                      sorted(stage1.items(), key=lambda kv: kv[1], reverse=True)[:K_CE]]
        rest = [sid for sid in ids if sid not in set(hybrid_top)]
        retrieved["hybrid-ce"] = ce_order(reranker, query, hybrid_top, id_to_index,
                                          user_docs, all_docs, dual=False) + rest
        retrieved["hybrid-ce-dual"] = ce_order(reranker, query, hybrid_top, id_to_index,
                                               user_docs, all_docs, dual=True) + rest

        for c in CONTROLS:
            m = metric_bundle(retrieved[c], correct, 5)
            m.update(metric_bundle(retrieved[c], correct, 10))
            for name, value in m.items():
                agg[c][name].append(value)
                by_type[c][qtype][name].append(value)

        if (qi + 1) % 50 == 0:
            el = time.time() - start
            print(f"  {qi+1}/{len(data)} ({(qi+1)/el:.2f} q/s)  " + "  ".join(
                f"{c}:R-all@5={np.mean(agg[c]['recall_all@5']):.3f}" for c in CONTROLS))

    el = time.time() - start
    out = {"k_ce": K_CE, "rows": len(data), "elapsed_sec": el, "controls": {}}
    print(f"\n{'='*100}")
    print(f"{'control':16s} {'R-any@5':>8s} {'R-all@5':>8s} {'NDCG@5':>8s} "
          f"{'R-any@10':>9s} {'R-all@10':>9s} {'NDCG@10':>8s}")
    for c in CONTROLS:
        m = {name: float(np.mean(v)) for name, v in agg[c].items()}
        out["controls"][c] = {
            "metrics": m,
            "by_type": {t: {name: float(np.mean(v)) for name, v in tm.items()}
                        for t, tm in by_type[c].items()},
        }
        print(f"{c:16s} {m['recall_any@5']*100:7.1f}% {m['recall_all@5']*100:7.1f}% "
              f"{m['ndcg_any@5']:8.4f} {m['recall_any@10']*100:8.1f}% "
              f"{m['recall_all@10']*100:8.1f}% {m['ndcg_any@10']:8.4f}")
    print(f"\nmemoria balanced (audited): R-any@5 98.1%  R-all@5 94.3%  "
          f"NDCG@5 0.9414  R-any@10 99.0%  R-all@10 97.1%  NDCG@10 0.9478")
    print(f"time: {el:.0f}s")

    for c in CONTROLS:
        print(f"\nby type — {c}:")
        for t in sorted(out["controls"][c]["by_type"]):
            tm = out["controls"][c]["by_type"][t]
            print(f"  {t:30s} R-all@5={tm['recall_all@5']:.3f} "
                  f"R-all@10={tm['recall_all@10']:.3f} NDCG@10={tm['ndcg_any@10']:.3f}")

    out_path = Path(__file__).parent / "results_controls_ce.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
