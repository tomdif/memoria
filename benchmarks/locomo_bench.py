"""
LoCoMo benchmark: multi-hop reasoning across long conversations.

Dataset: 10 conversations, 19-32 sessions each, 1986 QA pairs.
Evidence format: "D1:3" = session 1, dialog line 3.
We measure session-level retrieval: can we find the right session(s)?

Categories:
  1 - Single-fact
  2 - Single-fact temporal
  3 - Multi-fact
  4 - Multi-fact temporal
  5 - Open-ended / adversarial
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
import time
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from memoria.retriever import Retriever, RetrievalMode
from memoria.embeddings import Embedder
from memoria.graph import KnowledgeGraph
from memoria.schema import SCHEMA_SQL

CAT_NAMES = {
    1: "single-fact",
    2: "single-fact-temporal",
    3: "multi-fact",
    4: "multi-fact-temporal",
    5: "open-ended",
}


def recall_at_k(retrieved, correct, k):
    return 1.0 if correct.issubset(set(retrieved[:k])) else 0.0


def dcg(rels, k):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def ndcg_at_k(retrieved, correct, k):
    rels = [1.0 if r in correct else 0.0 for r in retrieved[:k]]
    idcg = dcg(sorted(rels, reverse=True), k)
    return dcg(rels, k) / idcg if idcg > 0 else 0.0


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
    args = parser.parse_args()

    if args.data is None:
        args.data = str(download())

    data = json.loads(Path(args.data).read_text())
    print(f"Loaded {len(data)} conversations, {sum(len(c['qa']) for c in data)} QA pairs")

    embedder = Embedder()
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA_SQL)
    kg = KnowledgeGraph(db)
    retriever = Retriever(kg, embedder)

    results = {}

    for mode in [RetrievalMode.SPEED, RetrievalMode.BALANCED, RetrievalMode.QUALITY]:
        r5_all, r10_all, n10_all = [], [], []
        cat_metrics = defaultdict(lambda: {"r5": [], "r10": [], "n10": []})
        t0 = time.time()
        skipped = 0

        for conv in data:
            session_ids, session_docs = build_session_docs(conv["conversation"])
            if not session_docs:
                continue

            # Pre-warm cache
            retriever._emb_cache.get_batch(session_docs)

            for qa in conv["qa"]:
                correct = extract_evidence_sessions(qa["evidence"])
                if not correct:
                    skipped += 1
                    continue

                ranked = retriever.retrieve_sessions(
                    qa["question"], session_docs, session_ids,
                    top_k=10, mode=mode,
                )
                ret = [sid for sid, _ in ranked]

                r5 = recall_at_k(ret, correct, 5)
                r10 = recall_at_k(ret, correct, 10)
                n10 = ndcg_at_k(ret, correct, 10)
                r5_all.append(r5)
                r10_all.append(r10)
                n10_all.append(n10)

                cat = qa["category"]
                cat_metrics[cat]["r5"].append(r5)
                cat_metrics[cat]["r10"].append(r10)
                cat_metrics[cat]["n10"].append(n10)

        elapsed = time.time() - t0
        n_total = len(r5_all)

        print(f"\n{'='*70}")
        print(f"LoCoMo — {mode.value} mode ({n_total} questions, {skipped} skipped)")
        print(f"{'='*70}")
        print(f"R@5:    {np.mean(r5_all)*100:.1f}%")
        print(f"R@10:   {np.mean(r10_all)*100:.1f}%")
        print(f"NDCG@10: {np.mean(n10_all):.4f}")
        print(f"Time:   {elapsed:.1f}s ({n_total/elapsed:.1f} q/s)")
        print(f"\nBy category:")
        for cat in sorted(cat_metrics):
            m = cat_metrics[cat]
            name = CAT_NAMES.get(cat, f"cat-{cat}")
            print(f"  {name:25s}  R@5={np.mean(m['r5']):.3f}  R@10={np.mean(m['r10']):.3f}  "
                  f"NDCG={np.mean(m['n10']):.3f}  (n={len(m['r5'])})")

        results[mode.value] = {
            "recall_5": float(np.mean(r5_all)),
            "recall_10": float(np.mean(r10_all)),
            "ndcg_10": float(np.mean(n10_all)),
            "total": n_total,
            "skipped": skipped,
            "time_sec": elapsed,
            "qps": n_total / elapsed,
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
