"""Legacy LongMemEval experiment runner.

This predates the audited metric and scope corrections. Its outputs are kept as
historical development artifacts and are not official-compatible benchmark
results. Use ``longmemeval_final.py`` for current measurements.

Evaluates retrieval quality: given a question and chat history,
can memoria find the sessions that contain the answer?

Modes:
  1. raw_vector  — baseline: embed sessions, cosine similarity (what mempalace does)
  2. kg_extract  — memoria's KG extraction + three-pass retrieval
  3. kg_spectral — KG + spectral compression: ingest all, compress, then search

Metrics: Recall@5, Recall@10, NDCG@10, broken down by question type.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memoria.schema import SCHEMA_SQL
from memoria.graph import KnowledgeGraph
from memoria.storage import ConversationStore
from memoria.embeddings import Embedder, cosine_similarity, top_k_similar
from memoria.extractor import ingest_extraction, extract_from_text
from memoria.retriever import Retriever
from memoria.spectral import build_adjacency, spectral_gap


# --- Metrics ---

def dcg(relevances: list[float], k: int) -> float:
    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        score += rel / math.log2(i + 2)
    return score


def ndcg_at_k(retrieved_ids: list[str], correct_ids: set[str], k: int) -> float:
    relevances = [1.0 if rid in correct_ids else 0.0 for rid in retrieved_ids[:k]]
    ideal = sorted(relevances, reverse=True)
    idcg = dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg(relevances, k) / idcg


def recall_at_k(retrieved_ids: list[str], correct_ids: set[str], k: int) -> float:
    """1.0 if ALL correct IDs appear in top-k, else 0.0."""
    retrieved_set = set(retrieved_ids[:k])
    return 1.0 if correct_ids.issubset(retrieved_set) else 0.0


# --- Session text extraction ---

def session_to_text(session: list[dict], user_only: bool = False) -> str:
    """Convert a session (list of turns) to plain text."""
    lines = []
    for turn in session:
        if user_only and turn.get("role") != "user":
            continue
        lines.append(turn.get("content", ""))
    return " ".join(lines)


# --- Mode 1: Raw vector baseline (what mempalace does) ---

def run_raw_vector(entry: dict, embedder: Embedder, top_k: int = 10) -> list[str]:
    """Embed each session as a document, query by cosine similarity."""
    sessions = entry["haystack_sessions"]
    session_ids = entry["haystack_session_ids"]

    # Build corpus: one document per session (user turns only, like mempalace)
    corpus_texts = []
    corpus_ids = []
    for sid, sess in zip(session_ids, sessions):
        text = session_to_text(sess, user_only=True)
        if text.strip():
            corpus_texts.append(text)
            corpus_ids.append(sid)

    if not corpus_texts:
        return []

    # Embed corpus and query
    corpus_embs = embedder.embed(corpus_texts)
    query_emb = embedder.embed_single(entry["question"])

    # Rank by similarity
    similar = top_k_similar(query_emb, corpus_embs, k=top_k)
    return [corpus_ids[idx] for idx, score in similar]


# --- Mode 2: Hybrid — vector retrieval + KG reranking ---

def run_hybrid(
    entry: dict,
    embedder: Embedder,
    llm_call=None,
    top_k: int = 10,
) -> list[str]:
    """Hybrid mode: vector search over sessions, reranked by entity overlap.

    1. Embed sessions and query (same as raw_vector)
    2. Extract entity names per session (fast, no embedding per entity)
    3. Extract entity names from query
    4. Sessions sharing entities with query get a graph bonus
    5. Spectral analysis of co-occurrence graph weights the bonus

    This is fast: only one batch embedding call (sessions), plus lightweight
    entity extraction per session.
    """
    sessions = entry["haystack_sessions"]
    session_ids = entry["haystack_session_ids"]

    # Phase 1: Build corpus + extract entity names per session
    corpus_texts = []
    corpus_ids = []
    session_entities = {}  # sid -> set of lowercased entity names

    for sid, sess in zip(session_ids, sessions):
        text = session_to_text(sess, user_only=True)
        if not text.strip():
            continue
        corpus_texts.append(text)
        corpus_ids.append(sid)

        # Fast entity extraction (just names, no KG, no embedding)
        extraction = _heuristic_extract(session_to_text(sess))
        session_entities[sid] = {e["name"].lower() for e in extraction.get("entities", [])}

    if not corpus_texts:
        return []

    # Phase 2: Embed sessions and query (one batch call)
    corpus_embs = embedder.embed(corpus_texts)
    query_emb = embedder.embed_single(entry["question"])

    # Vector scores
    similar = top_k_similar(query_emb, corpus_embs, k=min(top_k * 3, len(corpus_embs)))
    vector_scores = {corpus_ids[idx]: float(score) for idx, score in similar}

    # Phase 3: Query entity extraction
    query_extraction = _heuristic_extract(entry["question"])
    query_entity_names = {e["name"].lower() for e in query_extraction.get("entities", [])}

    # Also add important nouns from the query as pseudo-entities
    import re
    for word in re.findall(r"\b[A-Za-z]{3,}\b", entry["question"]):
        if word[0].isupper():
            query_entity_names.add(word.lower())

    # Phase 4: Entity overlap bonus
    entity_bonus = {}
    if query_entity_names:
        for sid, ent_names in session_entities.items():
            overlap = len(ent_names & query_entity_names)
            if overlap > 0:
                entity_bonus[sid] = overlap / len(query_entity_names)

    # Phase 5: Build co-occurrence graph and compute spectral weight
    # Entity co-occurrence: two entities that appear in the same session are linked
    spectral_weight = 0.15  # default modest weight
    if len(session_entities) >= 3:
        # Build entity co-occurrence adjacency
        all_ent_names = set()
        for names in session_entities.values():
            all_ent_names.update(names)
        if len(all_ent_names) >= 5:
            ent_list = sorted(all_ent_names)
            ent_idx = {name: i for i, name in enumerate(ent_list)}
            n_ents = len(ent_list)
            cooc_triples = []
            for sid, names in session_entities.items():
                name_list = sorted(names)
                for i in range(len(name_list)):
                    for j in range(i + 1, min(i + 5, len(name_list))):  # cap pairwise
                        cooc_triples.append({
                            "subject_id": name_list[i],
                            "object_id": name_list[j],
                            "relation_type": "fact",
                            "confidence": 1.0,
                        })
            if cooc_triples:
                A = build_adjacency(cooc_triples, ent_idx)
                gap, _ = spectral_gap(A)
                # Higher gap = more structured entity space = trust entity overlap more
                spectral_weight = min(gap * 3.0, 0.4)

    # Phase 6: Combine
    combined = {}
    all_sids = set(vector_scores.keys()) | set(entity_bonus.keys())
    for sid in all_sids:
        vs = vector_scores.get(sid, 0.0)
        eb = entity_bonus.get(sid, 0.0)
        combined[sid] = vs * (1.0 - spectral_weight) + eb * spectral_weight

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [sid for sid, score in ranked[:top_k]]


# --- Mode 3: KG + spectral compression ---

def run_kg_compressed(
    entry: dict,
    embedder: Embedder,
    llm_call=None,
    top_k: int = 10,
) -> list[str]:
    """Spectral compression mode: use entity co-occurrence graph to identify
    the most structurally important entities, then boost sessions containing them.

    Tests whether spectral compression preserves retrieval quality.
    """
    sessions = entry["haystack_sessions"]
    session_ids = entry["haystack_session_ids"]

    corpus_texts = []
    corpus_ids = []
    session_entities = {}

    for sid, sess in zip(session_ids, sessions):
        text = session_to_text(sess, user_only=True)
        if not text.strip():
            continue
        corpus_texts.append(text)
        corpus_ids.append(sid)
        extraction = _heuristic_extract(session_to_text(sess))
        session_entities[sid] = {e["name"].lower() for e in extraction.get("entities", [])}

    if not corpus_texts:
        return []

    # Build co-occurrence graph
    all_ent_names = set()
    for names in session_entities.values():
        all_ent_names.update(names)

    # Spectral importance of entities via co-occurrence graph
    important_entities = set()
    if len(all_ent_names) >= 5:
        ent_list = sorted(all_ent_names)
        ent_idx = {name: i for i, name in enumerate(ent_list)}
        cooc_triples = []
        for sid, names in session_entities.items():
            name_list = sorted(names)
            for i in range(len(name_list)):
                for j in range(i + 1, min(i + 5, len(name_list))):
                    cooc_triples.append({
                        "subject_id": name_list[i],
                        "object_id": name_list[j],
                        "relation_type": "fact",
                        "confidence": 1.0,
                    })
        if cooc_triples:
            from memoria.spectral import graph_laplacian
            from scipy.sparse.linalg import eigsh as _eigsh
            A = build_adjacency(cooc_triples, ent_idx)
            L = graph_laplacian(A)
            n = L.shape[0]
            k = min(6, n - 1)
            try:
                eigenvalues, eigenvectors = _eigsh(L, k=k, which="SM", tol=1e-6)
                idx_sort = np.argsort(np.real(eigenvalues))
                eigenvectors = np.real(eigenvectors[:, idx_sort])
                eigenvalues = np.real(eigenvalues[idx_sort])

                # Node importance from eigenvectors
                node_imp = np.zeros(n)
                for i in range(1, len(eigenvalues)):
                    lam = max(eigenvalues[i], 1e-10)
                    node_imp += (1.0 / lam) * eigenvectors[:, i] ** 2
                if node_imp.max() > 0:
                    node_imp /= node_imp.max()

                # Top 30% of entities by spectral importance
                threshold = np.percentile(node_imp, 70)
                for name, i in ent_idx.items():
                    if node_imp[i] >= threshold:
                        important_entities.add(name)
            except Exception:
                pass

    # Vector search
    corpus_embs = embedder.embed(corpus_texts)
    query_emb = embedder.embed_single(entry["question"])
    similar = top_k_similar(query_emb, corpus_embs, k=min(top_k * 3, len(corpus_embs)))
    vector_scores = {corpus_ids[idx]: float(score) for idx, score in similar}

    # Spectral bonus: sessions with spectrally-important entities
    spectral_bonus = {}
    if important_entities:
        for sid, ent_names in session_entities.items():
            overlap = len(ent_names & important_entities)
            if overlap > 0:
                spectral_bonus[sid] = overlap / max(len(important_entities), 1) * 5

    # Combine
    combined = {}
    for sid in set(vector_scores) | set(spectral_bonus):
        combined[sid] = vector_scores.get(sid, 0.0) * 0.85 + spectral_bonus.get(sid, 0.0) * 0.15

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [sid for sid, score in ranked[:top_k]]


# --- Heuristic extraction fallback ---

def _heuristic_extract(text: str) -> dict:
    """Quick regex extraction for benchmark speed."""
    import re
    entities = []
    facts = []

    # Capitalized phrases
    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
        name = match.group(1)
        if len(name) > 2:
            entities.append({"name": name, "type": "concept"})

    # Quoted strings
    for match in re.finditer(r'"([^"]+)"', text):
        entities.append({"name": match.group(1), "type": "concept"})

    # Simple relation patterns
    for match in re.finditer(
        r"\b(\w+(?:\s+\w+)?)\s+(use[sd]?|is|was|prefer[sd]?|like[sd]?|chose|switched to)\s+(\w+(?:\s+\w+)?)\b",
        text, re.IGNORECASE,
    ):
        facts.append({
            "subject": match.group(1),
            "predicate": match.group(2).lower().replace(" ", "_"),
            "object": match.group(3),
            "confidence": 0.5,
        })

    # Deduplicate
    seen = set()
    unique = []
    for e in entities:
        if e["name"].lower() not in seen:
            seen.add(e["name"].lower())
            unique.append(e)

    return {"entities": unique[:50], "facts": facts[:30], "causal": [], "decisions": []}


# --- Benchmark runner ---

@dataclass
class BenchmarkResult:
    mode: str
    total: int = 0
    recall_5: float = 0.0
    recall_10: float = 0.0
    ndcg_10: float = 0.0
    by_type: dict = field(default_factory=dict)
    elapsed_sec: float = 0.0
    errors: int = 0


def run_benchmark(
    data_path: str | Path,
    modes: list[str] = None,
    max_questions: int | None = None,
    llm_call=None,
    model_name: str = "all-MiniLM-L6-v2",
    verbose: bool = True,
) -> dict[str, BenchmarkResult]:
    """Run LongMemEval benchmark.

    Args:
        data_path: Path to longmemeval_s_cleaned.json
        modes: List of modes to test: "raw_vector", "hybrid", "kg_spectral"
        max_questions: Limit number of questions (for quick testing)
        llm_call: Optional LLM for extraction (None = heuristic only)
        model_name: Sentence transformer model
        verbose: Print progress

    Returns: {mode_name: BenchmarkResult}
    """
    if modes is None:
        modes = ["raw_vector", "hybrid"]

    data = json.loads(Path(data_path).read_text())
    if max_questions:
        data = data[:max_questions]

    embedder = Embedder(model_name)

    mode_fns = {
        "raw_vector": lambda e: run_raw_vector(e, embedder),
        "hybrid": lambda e: run_hybrid(e, embedder, llm_call),
        "kg_compressed": lambda e: run_kg_compressed(e, embedder, llm_call),
    }

    results = {}

    for mode in modes:
        if mode not in mode_fns:
            print(f"Unknown mode: {mode}")
            continue

        fn = mode_fns[mode]
        br = BenchmarkResult(mode=mode)
        type_metrics = defaultdict(lambda: {"recall_5": [], "recall_10": [], "ndcg_10": []})

        start = time.time()

        for i, entry in enumerate(data):
            qid = entry["question_id"]
            qtype = entry["question_type"]
            correct = set(entry["answer_session_ids"])

            try:
                retrieved = fn(entry)
            except Exception as e:
                if verbose:
                    print(f"  ERROR q{i} ({qid}): {e}")
                br.errors += 1
                continue

            r5 = recall_at_k(retrieved, correct, 5)
            r10 = recall_at_k(retrieved, correct, 10)
            n10 = ndcg_at_k(retrieved, correct, 10)

            type_metrics[qtype]["recall_5"].append(r5)
            type_metrics[qtype]["recall_10"].append(r10)
            type_metrics[qtype]["ndcg_10"].append(n10)

            br.total += 1

            if verbose and (i + 1) % 10 == 0:
                running_r5 = np.mean([v for tm in type_metrics.values() for v in tm["recall_5"]])
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                eta = (len(data) - i - 1) / rate if rate > 0 else 0
                print(
                    f"  [{mode}] {i+1}/{len(data)} "
                    f"R@5={running_r5:.3f} "
                    f"({rate:.1f} q/s, ETA {eta:.0f}s)"
                )

        br.elapsed_sec = time.time() - start

        # Aggregate metrics
        all_r5 = [v for tm in type_metrics.values() for v in tm["recall_5"]]
        all_r10 = [v for tm in type_metrics.values() for v in tm["recall_10"]]
        all_n10 = [v for tm in type_metrics.values() for v in tm["ndcg_10"]]

        br.recall_5 = float(np.mean(all_r5)) if all_r5 else 0.0
        br.recall_10 = float(np.mean(all_r10)) if all_r10 else 0.0
        br.ndcg_10 = float(np.mean(all_n10)) if all_n10 else 0.0

        br.by_type = {}
        for qtype, metrics in type_metrics.items():
            br.by_type[qtype] = {
                "recall_5": float(np.mean(metrics["recall_5"])) if metrics["recall_5"] else 0.0,
                "recall_10": float(np.mean(metrics["recall_10"])) if metrics["recall_10"] else 0.0,
                "ndcg_10": float(np.mean(metrics["ndcg_10"])) if metrics["ndcg_10"] else 0.0,
                "count": len(metrics["recall_5"]),
            }

        results[mode] = br

    return results


def print_results(results: dict[str, BenchmarkResult]):
    """Pretty-print benchmark results."""
    print("\n" + "=" * 70)
    print("LONGMEMEVAL BENCHMARK RESULTS")
    print("=" * 70)

    for mode, br in results.items():
        print(f"\n--- {mode} ---")
        print(f"Questions: {br.total} | Errors: {br.errors} | Time: {br.elapsed_sec:.1f}s")
        print(f"R@5:  {br.recall_5:.4f} ({br.recall_5*100:.1f}%)")
        print(f"R@10: {br.recall_10:.4f} ({br.recall_10*100:.1f}%)")
        print(f"NDCG@10: {br.ndcg_10:.4f}")
        print(f"\nBy question type:")
        for qtype in sorted(br.by_type.keys()):
            m = br.by_type[qtype]
            print(f"  {qtype:30s}  R@5={m['recall_5']:.3f}  R@10={m['recall_10']:.3f}  NDCG@10={m['ndcg_10']:.3f}  (n={m['count']})")

    # Comparison table
    if len(results) > 1:
        print(f"\n{'='*70}")
        print(f"{'Mode':20s} {'R@5':>8s} {'R@10':>8s} {'NDCG@10':>8s} {'Time':>8s}")
        print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for mode, br in results.items():
            print(f"{mode:20s} {br.recall_5*100:7.1f}% {br.recall_10*100:7.1f}% {br.ndcg_10:8.4f} {br.elapsed_sec:7.1f}s")
        # Reference
        print(f"{'mempalace (raw)':20s} {'96.6':>7s}% {'94.8':>7s}% {'---':>8s} {'---':>8s}")


# --- CLI ---

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run LongMemEval benchmark on memoria")
    parser.add_argument("--data", default=None, help="Path to longmemeval_s_cleaned.json")
    parser.add_argument("--modes", nargs="+", default=["raw_vector", "hybrid"],
                        help="Modes to test")
    parser.add_argument("--max", type=int, default=None, help="Max questions (for quick test)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", default=None, help="Save results as JSON")
    args = parser.parse_args()

    # Find or download data
    if args.data:
        data_path = args.data
    else:
        from download_data import download
        data_path = download()

    results = run_benchmark(
        data_path=data_path,
        modes=args.modes,
        max_questions=args.max,
        model_name=args.model,
        verbose=not args.quiet,
    )

    print_results(results)

    if args.output:
        out = {}
        for mode, br in results.items():
            out[mode] = {
                "recall_5": br.recall_5,
                "recall_10": br.recall_10,
                "ndcg_10": br.ndcg_10,
                "total": br.total,
                "errors": br.errors,
                "elapsed_sec": br.elapsed_sec,
                "by_type": br.by_type,
            }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
