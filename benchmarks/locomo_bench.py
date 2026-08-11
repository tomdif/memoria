"""LoCoMo session-level evidence-retrieval diagnostic.

Dataset: 10 conversations, 19-32 sessions each, 1986 QA pairs.
Evidence format: "D1:3" = session 1, dialog line 3.
We measure whether Memoria ranks the annotated evidence session(s).  LoCoMo's
official task is answer generation scored with token F1, so these retrieval
numbers are a diagnostic and must not be compared directly to QA scores.

Categories:
  1 - Single-hop
  2 - Temporal
  3 - Multi-hop
  4 - Open-domain knowledge
  5 - Adversarial
"""

from __future__ import annotations

import json
import sqlite3
import sys
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from memoria.retriever import Retriever, RetrievalMode
from memoria.embeddings import Embedder
from memoria.graph import KnowledgeGraph
from memoria.schema import SCHEMA_SQL
try:
    from .retrieval_metrics import metric_bundle, rank_dense, rank_flat_bm25
except ImportError:  # Direct execution: python benchmarks/locomo_bench.py
    from retrieval_metrics import metric_bundle, rank_dense, rank_flat_bm25

CAT_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}

WINDOW_TURNS = 4
WINDOW_STRIDE = 2


def extract_evidence_sessions(evidence: list[str]) -> set[str]:
    """Extract session IDs from evidence like ['D1:3', 'D5:12']."""
    sessions = set()
    for ev in evidence:
        m = re.match(r"D(\d+)", ev)
        if m:
            sessions.add(f"session_{m.group(1)}")
    return sessions


def build_session_docs(conversation: dict) -> tuple[list[str], list[str]]:
    """Build session documents from a LoCoMo conversation."""
    session_keys = sorted(
        [k for k in conversation if k.startswith("session_") and not k.endswith("date_time")],
        key=lambda x: int(x.split("_")[1]),
    )
    session_ids = []
    session_docs = []
    for key in session_keys:
        turns = conversation[key]
        if not isinstance(turns, list):
            continue
        text = "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)
        if text.strip():
            session_ids.append(key)
            session_docs.append(text)
    return session_ids, session_docs


def _render_turn(turn: dict) -> str:
    """Render all text supplied by LoCoMo's official text-only protocol."""
    parts = [f"{turn['speaker']}: {turn['text']}"]
    caption = turn.get("blip_caption")
    if caption:
        parts.append(f"Image caption: {caption}")
    return "\n".join(parts)


def build_session_windows(
    conversation: dict,
    window_turns: int = WINDOW_TURNS,
    stride: int = WINDOW_STRIDE,
) -> tuple[list[str], list[str]]:
    """Build overlapping, date-aware windows aligned to parent sessions."""
    if window_turns <= 0 or stride <= 0:
        raise ValueError("window_turns and stride must be positive")

    session_ids, _ = build_session_docs(conversation)
    chunk_docs: list[str] = []
    group_ids: list[str] = []
    for session_id in session_ids:
        turns = conversation[session_id]
        date = conversation.get(f"{session_id}_date_time", "unknown date")
        rendered_turns = [_render_turn(turn) for turn in turns]
        last_start = max(0, len(rendered_turns) - window_turns)
        starts = list(range(0, last_start + 1, stride))
        if not starts or starts[-1] != last_start:
            starts.append(last_start)

        seen_windows = set()
        for start in starts:
            window = tuple(rendered_turns[start:start + window_turns])
            if not window or window in seen_windows:
                continue
            seen_windows.add(window)
            chunk_docs.append(
                f"Session date: {date}\n" + "\n".join(window)
            )
            group_ids.append(session_id)

    return chunk_docs, group_ids


def download():
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "locomo10.json"
    if path.exists():
        return path
    import urllib.request
    url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
    print(f"Downloading LoCoMo from {url}...")
    urllib.request.urlretrieve(url, path)
    print(f"  Downloaded to {path}")
    return path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--output", default="results_locomo.json")
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Run a deterministic conversation-level subset for diagnostics",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in RetrievalMode] + ["all"],
        default=RetrievalMode.BALANCED.value,
    )
    parser.add_argument(
        "--retriever",
        choices=["memoria", "flat-bm25", "minilm"],
        default="memoria",
        help="System under test or an apples-to-apples local baseline",
    )
    parser.add_argument(
        "--profile",
        choices=["windowed", "legacy"],
        default="windowed",
        help="Memoria indexing strategy; baselines always retain session indexing",
    )
    args = parser.parse_args()

    if args.data is None:
        args.data = str(download())

    data = json.loads(Path(args.data).read_text())
    if args.max_conversations is not None:
        data = data[:args.max_conversations]
    print(f"Loaded {len(data)} conversations, {sum(len(c['qa']) for c in data)} QA pairs")

    embedder = Embedder()
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA_SQL)
    kg = KnowledgeGraph(db)
    retriever = Retriever(kg, embedder)

    results = {}

    modes = (
        list(RetrievalMode)
        if args.mode == "all"
        else [RetrievalMode(args.mode)]
    )
    metric_names = [
        "recall_any@5", "recall_all@5", "ndcg_any@5",
        "recall_any@10", "recall_all@10", "ndcg_any@10",
    ]

    for mode in modes:
        all_metrics = {name: [] for name in metric_names}
        core_metrics = {name: [] for name in metric_names}
        cat_metrics = defaultdict(lambda: {name: [] for name in metric_names})
        t0 = time.time()
        skipped = 0

        for conv in data:
            session_ids, session_docs = build_session_docs(conv["conversation"])
            if not session_docs:
                continue

            chunk_docs, chunk_group_ids = build_session_windows(conv["conversation"])

            if args.retriever != "flat-bm25":
                retriever._emb_cache.get_batch(
                    chunk_docs
                    if args.retriever == "memoria" and args.profile == "windowed"
                    else session_docs
                )

            for qa in conv["qa"]:
                correct = extract_evidence_sessions(qa["evidence"])
                if not correct:
                    skipped += 1
                    continue

                if args.retriever == "memoria":
                    if args.profile == "windowed":
                        ranked = retriever.retrieve_grouped_sessions(
                            qa["question"], chunk_docs, chunk_group_ids,
                            top_k=10, mode=mode,
                        )
                    else:
                        ranked = retriever.retrieve_sessions(
                            qa["question"], session_docs, session_ids,
                            top_k=10, mode=mode,
                        )
                    ret = [sid for sid, _ in ranked]
                elif args.retriever == "flat-bm25":
                    ret = rank_flat_bm25(qa["question"], session_docs, session_ids, 10)
                else:
                    document_embeddings = retriever._emb_cache.get_batch(session_docs)
                    query_embedding = embedder.embed_single(qa["question"])
                    ret = rank_dense(
                        query_embedding, document_embeddings, session_ids, 10
                    )

                cat = qa["category"]
                metrics = metric_bundle(ret, correct, 5)
                metrics.update(metric_bundle(ret, correct, 10))
                for name, value in metrics.items():
                    all_metrics[name].append(value)
                    cat_metrics[cat][name].append(value)
                    if cat != 5:
                        core_metrics[name].append(value)

                if len(all_metrics["recall_all@5"]) % 250 == 0:
                    print(
                        f"  {len(all_metrics['recall_all@5'])} scored "
                        f"R-all@5={np.mean(all_metrics['recall_all@5']):.3f}"
                    )

        elapsed = time.time() - t0
        n_total = len(all_metrics["recall_all@5"])
        n_core = len(core_metrics["recall_all@5"])

        print(f"\n{'='*70}")
        system_label = (
            f"memoria-{mode.value}-{args.profile}"
            if args.retriever == "memoria" else args.retriever
        )
        print(f"LoCoMo retrieval diagnostic — {system_label}")
        print(f"{'='*70}")
        print(f"Core rows:       {n_core} (categories 1-4)")
        print(f"All scored rows: {n_total} ({skipped} without evidence skipped)")
        print(f"Core Recall-All@5:  {np.mean(core_metrics['recall_all@5'])*100:.1f}%")
        print(f"Core Recall-All@10: {np.mean(core_metrics['recall_all@10'])*100:.1f}%")
        print(f"Core NDCG-Any@10:   {np.mean(core_metrics['ndcg_any@10']):.4f}")
        print(f"All Recall-All@5:   {np.mean(all_metrics['recall_all@5'])*100:.1f}%")
        print(f"All Recall-All@10:  {np.mean(all_metrics['recall_all@10'])*100:.1f}%")
        print(f"Time:   {elapsed:.1f}s ({n_total/elapsed:.1f} q/s)")
        print(f"\nBy category:")
        for cat in sorted(cat_metrics):
            m = cat_metrics[cat]
            name = CAT_NAMES.get(cat, f"cat-{cat}")
            print(f"  {name:25s}  R-all@5={np.mean(m['recall_all@5']):.3f}  "
                  f"R-all@10={np.mean(m['recall_all@10']):.3f}  "
                  f"NDCG@10={np.mean(m['ndcg_any@10']):.3f}  "
                  f"(n={len(m['recall_all@5'])})")

        results[mode.value] = {
            "benchmark": "LoCoMo session evidence retrieval diagnostic",
            "core_four_categories": {
                "total": n_core,
                "metrics": {
                    name: float(np.mean(values))
                    for name, values in core_metrics.items()
                },
            },
            "all_categories": {
                "total": n_total,
                "metrics": {
                    name: float(np.mean(values))
                    for name, values in all_metrics.items()
                },
            },
            "skipped": skipped,
            "time_sec": elapsed,
            "qps": n_total / elapsed,
            "retriever": args.retriever,
            "profile": args.profile if args.retriever == "memoria" else "session",
            "by_category": {
                CAT_NAMES.get(cat, f"cat-{cat}"): {k: float(np.mean(v)) for k, v in m.items()}
                for cat, m in cat_metrics.items()
            },
        }

    out = Path(__file__).parent / args.output
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")
    db.close()


if __name__ == "__main__":
    main()
