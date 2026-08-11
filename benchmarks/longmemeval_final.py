"""LongMemEval-S session-retrieval benchmark for Memoria.

The default scope mirrors the official retrieval harness: abstention instances
and instances without a user-side evidence label are excluded, and the reported
metrics are binary recall-all plus the benchmark's NDCG implementation.  This is
a retrieval evaluation, not the benchmark's end-to-end, LLM-judged QA score.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from memoria.embeddings import Embedder
from memoria.graph import KnowledgeGraph
from memoria.retriever import Retriever, RetrievalMode
from memoria.schema import SCHEMA_SQL
try:
    from .retrieval_metrics import metric_bundle, rank_dense, rank_flat_bm25
except ImportError:  # Direct execution: python benchmarks/longmemeval_final.py
    from retrieval_metrics import metric_bundle, rank_dense, rank_flat_bm25


def user_evidence_session_ids(question: dict) -> set[str]:
    """Return sessions with an official user-side ``has_answer`` label."""
    return {
        session_id
        for session_id, session in zip(
            question["haystack_session_ids"], question["haystack_sessions"]
        )
        if any(
            turn.get("role") == "user" and turn.get("has_answer") is True
            for turn in session
        )
    }


def is_official_retrieval_instance(question: dict) -> bool:
    """Apply the exclusions in LongMemEval's official retrieval harness."""
    return (
        "_abs" not in question["question_id"]
        and bool(user_evidence_session_ids(question))
    )


def build_session_documents(q):
    """Return aligned IDs, user-only documents, and all-turn documents."""
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
    return ids, user_docs, all_docs


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--scope",
        choices=["official", "non-abstention"],
        default="official",
        help=(
            "official excludes abstention and rows without user-side evidence; "
            "non-abstention evaluates every answer-bearing non-abstention row"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in RetrievalMode],
        default=RetrievalMode.BALANCED.value,
    )
    parser.add_argument(
        "--retriever",
        choices=["memoria", "flat-bm25", "minilm"],
        default="memoria",
        help="System under test or an apples-to-apples local baseline",
    )
    args = parser.parse_args()

    if args.data is None:
        from download_data import download
        args.data = str(download())

    source_data = json.loads(Path(args.data).read_text())
    abstention_count = sum("_abs" in q["question_id"] for q in source_data)
    no_user_target_count = sum(
        "_abs" not in q["question_id"] and not user_evidence_session_ids(q)
        for q in source_data
    )
    if args.scope == "official":
        data = [q for q in source_data if is_official_retrieval_instance(q)]
    else:
        data = [q for q in source_data if "_abs" not in q["question_id"]]
    if args.max:
        data = data[: args.max]

    embedder = Embedder()
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA_SQL)
    retriever = Retriever(KnowledgeGraph(db), embedder)
    mode = RetrievalMode(args.mode)

    metric_names = [
        "recall_any@5", "recall_all@5", "ndcg_any@5",
        "recall_any@10", "recall_all@10", "ndcg_any@10",
    ]
    all_metrics = {name: [] for name in metric_names}
    type_metrics = defaultdict(lambda: {name: [] for name in metric_names})
    start = time.time()

    for i, q in enumerate(data):
        correct = (
            user_evidence_session_ids(q)
            if args.scope == "official"
            else set(q["answer_session_ids"])
        )
        ids, user_docs, all_docs = build_session_documents(q)
        if args.retriever == "memoria":
            ranked = retriever.retrieve_sessions(
                q["question"],
                user_docs,
                ids,
                top_k=10,
                mode=mode,
                all_docs=all_docs,
            )
            retrieved = [session_id for session_id, _ in ranked]
        elif args.retriever == "flat-bm25":
            retrieved = rank_flat_bm25(q["question"], user_docs, ids, 10)
        else:
            document_embeddings = retriever._emb_cache.get_batch(user_docs)
            query_embedding = embedder.embed_single(q["question"])
            retrieved = rank_dense(query_embedding, document_embeddings, ids, 10)

        metrics = metric_bundle(retrieved, correct, 5)
        metrics.update(metric_bundle(retrieved, correct, 10))
        qtype = q["question_type"]
        for name, value in metrics.items():
            all_metrics[name].append(value)
            type_metrics[qtype][name].append(value)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(
                f"  {i+1}/{len(data)} Recall-All@5="
                f"{np.mean(all_metrics['recall_all@5']):.3f} "
                f"Recall-All@10={np.mean(all_metrics['recall_all@10']):.3f} "
                f"({(i+1)/elapsed:.1f} q/s)"
            )

    elapsed = time.time() - start
    fails = sum(value == 0 for value in all_metrics["recall_all@5"])
    impossible = sum(
        len(
            user_evidence_session_ids(q)
            if args.scope == "official"
            else q["answer_session_ids"]
        ) > 5
        for q in data
    )

    print(f"\n{'='*75}")
    system_label = (
        f"memoria-{mode.value}" if args.retriever == "memoria" else args.retriever
    )
    print(f"LongMemEval-S retrieval — {system_label} ({args.scope})")
    print(f"{'='*75}")
    print(f"Rows:          {len(data)} scored / {len(source_data)} total")
    print(f"Excluded:      {abstention_count} abstention, "
          f"{no_user_target_count} without user-side evidence")
    print(f"Recall-Any@5:  {np.mean(all_metrics['recall_any@5'])*100:.1f}%")
    print(f"Recall-All@5:  {np.mean(all_metrics['recall_all@5'])*100:.1f}% "
          f"({fails} failures, {impossible} impossible at k=5)")
    print(f"NDCG-Any@5:    {np.mean(all_metrics['ndcg_any@5']):.4f}")
    print(f"Recall-Any@10: {np.mean(all_metrics['recall_any@10'])*100:.1f}%")
    print(f"Recall-All@10: {np.mean(all_metrics['recall_all@10'])*100:.1f}%")
    print(f"NDCG-Any@10:   {np.mean(all_metrics['ndcg_any@10']):.4f}")
    print(f"Time:   {elapsed:.0f}s ({len(data)/elapsed:.1f} q/s)")
    print(f"\nBy type:")
    for t in sorted(type_metrics):
        m = type_metrics[t]
        print(
            f"  {t:30s}  R-all@5={np.mean(m['recall_all@5']):.3f}  "
            f"R-all@10={np.mean(m['recall_all@10']):.3f}  "
            f"NDCG@10={np.mean(m['ndcg_any@10']):.3f}  "
            f"(n={len(m['recall_all@5'])})"
        )

    if args.output:
        out = {
            "benchmark": "LongMemEval-S session retrieval",
            "scope": args.scope,
            "metrics": {
                name: float(np.mean(values))
                for name, values in all_metrics.items()
            },
            "failures": fails,
            "impossible": impossible,
            "total": len(data),
            "source_total": len(source_data),
            "excluded_abstention": abstention_count,
            "excluded_without_user_evidence": no_user_target_count,
            "elapsed_sec": elapsed,
            "qps": len(data) / elapsed,
            "mode": mode.value,
            "retriever": args.retriever,
            "by_type": {
                t: {k: float(np.mean(v)) for k, v in m.items()}
                for t, m in type_metrics.items()
            },
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nSaved to {args.output}")

    db.close()


if __name__ == "__main__":
    main()
