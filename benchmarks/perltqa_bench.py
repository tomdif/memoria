"""Frozen-retriever audit on the official PerLTQA English v2 dataset.

PerLTQA's retrieval task asks a system to recover one referenced memory from a
character-specific bank.  This runner preserves the dataset's native memory
record as the scored unit, windows long event/dialogue records for retrieval,
and collapses windows back to unique record IDs before calculating the paper's
Recall@1/2/3/5 metrics.

The official v2 files contain a small number of QA references whose memory IDs
are absent from the supplied bank.  The primary strict scope counts those as
misses; a valid-gold scope is reported separately for data-quality diagnosis.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
import re
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from memoria.embeddings import Embedder
from memoria.graph import KnowledgeGraph
from memoria.retriever import Retriever, RetrievalMode
from memoria.schema import SCHEMA_SQL

try:
    from .download_perltqa import FILES, UPSTREAM_COMMIT, download
except ImportError:  # Direct execution: python benchmarks/perltqa_bench.py
    from download_perltqa import FILES, UPSTREAM_COMMIT, download


MEMORY_TYPES = ("profile", "social_relationship", "events", "dialogues")
WINDOW_ITEMS = 4
WINDOW_STRIDE = 2
PROFILE_ALIASES = {
    "Awards": "Awards and Role Models",
    "Role Models": "Awards and Role Models",
}


@dataclass(frozen=True)
class QuestionRecord:
    character: str
    memory_type: str
    question: str
    gold_id: str


@dataclass
class MemoryBank:
    parent_ids: list[str]
    parent_docs: list[str]
    chunk_docs: list[str]
    chunk_parent_ids: list[str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(_render(item) for item in value)
    if isinstance(value, dict):
        return "\n".join(f"{key}: {_render(item)}" for key, item in value.items())
    return html.unescape(str(value)).strip()


def _overlapping_windows(
    items: list[str], size: int = WINDOW_ITEMS, stride: int = WINDOW_STRIDE
) -> list[list[str]]:
    if size <= 0 or stride <= 0:
        raise ValueError("window size and stride must be positive")
    if not items:
        return []
    last_start = max(0, len(items) - size)
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return [items[start:start + size] for start in starts]


def _sentence_windows(text: str) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", _render(text))
        if sentence.strip()
    ]
    return [" ".join(window) for window in _overlapping_windows(sentences)]


def _record_id(memory_type: str, source_id: str) -> str:
    if memory_type not in MEMORY_TYPES:
        raise ValueError(f"Unknown memory type: {memory_type}")
    return f"{memory_type}:{source_id}"


def normalize_gold_id(gold_id: str, parent_ids: set[str]) -> str:
    """Resolve profile-field aliases against one character's native bank."""
    if gold_id in parent_ids:
        return gold_id
    memory_type, source_id = gold_id.split(":", 1)
    if memory_type == "profile" and source_id in PROFILE_ALIASES:
        fallback = _record_id(memory_type, PROFILE_ALIASES[source_id])
        if fallback in parent_ids:
            return fallback
    return gold_id


def parse_reference_ids(memory_type: str, raw_reference: str) -> list[str]:
    """Parse PerLTQA's profile labels or serialized list-string references."""
    if memory_type == "profile":
        references = [raw_reference]
    else:
        try:
            references = ast.literal_eval(raw_reference)
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"Malformed Reference Memory: {raw_reference!r}") from error
        if not isinstance(references, list):
            raise ValueError("Reference Memory must decode to a list")
    if not references or not all(isinstance(item, str) and item for item in references):
        raise ValueError("Reference Memory must contain non-empty string IDs")
    return [_record_id(memory_type, item) for item in references]


def load_character_questions(raw_qa: list[dict]) -> dict[str, dict]:
    characters: dict[str, dict] = {}
    for group in raw_qa:
        if not isinstance(group, dict) or len(group) != 1:
            raise ValueError("Each PerLTQA group must contain exactly one character")
        name, sections = next(iter(group.items()))
        if name in characters:
            raise ValueError(f"Duplicate QA character: {name}")
        if set(sections) != set(MEMORY_TYPES):
            raise ValueError(f"Unexpected memory sections for {name}")
        characters[name] = sections
    return characters


def _reference_prefixes(sections: dict) -> set[str]:
    prefixes: set[str] = set()
    for memory_type in MEMORY_TYPES[1:]:
        for group in sections[memory_type]:
            for questions in group.values():
                for row in questions:
                    for reference in parse_reference_ids(
                        memory_type, row["Reference Memory"]
                    ):
                        source_id = reference.split(":", 1)[1]
                        prefixes.add(source_id.split("_", 1)[0])
    return prefixes


def _memory_prefixes(memory: dict) -> set[str]:
    relationships = memory["social_relationship"]
    source_ids = (
        list(relationships) if isinstance(relationships, dict) else []
    ) + list(memory["events"])
    return {source_id.split("_", 1)[0] for source_id in source_ids}


def resolve_character_name(
    qa_name: str, sections: dict, memories: dict[str, dict]
) -> str:
    """Resolve translated key mismatches using unique structural ID prefixes."""
    if qa_name in memories:
        return qa_name
    prefixes = _reference_prefixes(sections)
    candidates = [
        memory_name
        for memory_name, memory in memories.items()
        if prefixes and prefixes <= _memory_prefixes(memory)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Cannot uniquely resolve QA character {qa_name!r}: {candidates}"
        )
    return candidates[0]


def iter_questions(character: str, sections: dict) -> Iterable[QuestionRecord]:
    for row in sections["profile"]:
        gold = parse_reference_ids("profile", row["Reference Memory"])
        if len(gold) != 1:
            raise ValueError("PerLTQA retrieval rows must have exactly one gold memory")
        yield QuestionRecord(character, "profile", row["Question"], gold[0])

    for memory_type in MEMORY_TYPES[1:]:
        for group in sections[memory_type]:
            for questions in group.values():
                for row in questions:
                    gold = parse_reference_ids(
                        memory_type, row["Reference Memory"]
                    )
                    if len(gold) != 1:
                        raise ValueError(
                            "PerLTQA retrieval rows must have exactly one gold memory"
                        )
                    yield QuestionRecord(
                        character, memory_type, row["Question"], gold[0]
                    )


def build_memory_bank(memory: dict) -> MemoryBank:
    """Render native memory records and aligned four-item retrieval windows."""
    protagonist = _render(memory["profile"].get("Protagonist", "character"))
    parent_ids: list[str] = []
    parent_docs: list[str] = []
    chunk_docs: list[str] = []
    chunk_parent_ids: list[str] = []

    def add(parent_id: str, document: str, chunks: list[str] | None = None) -> None:
        if parent_id in set(parent_ids):
            raise ValueError(f"Duplicate memory record ID: {parent_id}")
        clean_document = document.strip()
        if not clean_document:
            raise ValueError(f"Empty memory record: {parent_id}")
        parent_ids.append(parent_id)
        parent_docs.append(clean_document)
        for chunk in chunks or [clean_document]:
            if chunk.strip():
                chunk_docs.append(chunk.strip())
                chunk_parent_ids.append(parent_id)

    for field, value in memory["profile"].items():
        add(
            _record_id("profile", field),
            f"Profile for {protagonist}\n{field}: {_render(value)}",
        )
    add(
        _record_id("profile", "profile_description"),
        f"Profile summary for {protagonist}\n{_render(memory['profile_description'])}",
    )

    relationships = memory["social_relationship"]
    if isinstance(relationships, dict):
        for source_id, relationship in relationships.items():
            document = "\n".join(
                f"{field}: {_render(value)}" for field, value in relationship.items()
            )
            add(_record_id("social_relationship", source_id), document)
    elif isinstance(relationships, str) and relationships.strip():
        # Some v2 characters retain one unindexed relationship narrative. It
        # remains searchable, but cannot satisfy an indexed gold reference.
        add(
            _record_id("social_relationship", "unstructured"),
            _render(relationships),
        )
    else:
        raise ValueError("social_relationship must be an object or non-empty string")

    for source_id, event in memory["events"].items():
        metadata_fields = (
            "summary", "Characters", "Creation Time", "Last Accessed Time", "Theme"
        )
        metadata = "\n".join(
            f"{field}: {_render(event.get(field, ''))}" for field in metadata_fields
        )
        content = _render(event.get("content", ""))
        document = f"{metadata}\nContent: {content}"
        event_windows = _sentence_windows(content)
        chunks = [
            f"{metadata}\nContent: {window}" for window in event_windows
        ] or [document]
        add(_record_id("events", source_id), document, chunks)

    for source_id, dialogue in memory["dialogues"].items():
        related_event = _render(dialogue.get("events", ""))
        all_parts: list[str] = []
        chunks: list[str] = []
        for timestamp, turns in dialogue.get("contents", {}).items():
            rendered_turns = [_render(turn) for turn in turns]
            heading = f"Related event: {related_event}\nDate: {_render(timestamp)}"
            all_parts.append(f"{heading}\n" + "\n".join(rendered_turns))
            chunks.extend(
                f"{heading}\n" + "\n".join(window)
                for window in _overlapping_windows(rendered_turns)
            )
        if not all_parts:
            all_parts = [f"Related event: {related_event}"]
        add(
            _record_id("dialogues", source_id),
            "\n".join(all_parts),
            chunks or all_parts,
        )

    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("Memory record IDs must be unique")
    if len(chunk_docs) != len(chunk_parent_ids):
        raise ValueError("Chunk documents and parent IDs must align")
    return MemoryBank(parent_ids, parent_docs, chunk_docs, chunk_parent_ids)


def _collapse_scores(
    scores: np.ndarray, parent_ids: list[str], top_k: int
) -> list[str]:
    best: dict[str, float] = {}
    for score, parent_id in zip(np.asarray(scores).reshape(-1), parent_ids):
        best[parent_id] = max(best.get(parent_id, -np.inf), float(score))
    return [
        parent_id for parent_id, _ in
        sorted(best.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    ]


def rank_window_bm25(
    query: str, chunk_docs: list[str], chunk_parent_ids: list[str], top_k: int
) -> list[str]:
    from rank_bm25 import BM25Okapi

    tokenize = lambda value: re.findall(r"\w+", value.lower())
    model = BM25Okapi([tokenize(document) for document in chunk_docs])
    return _collapse_scores(
        np.asarray(model.get_scores(tokenize(query)), dtype=float),
        chunk_parent_ids,
        top_k,
    )


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-data", default=None)
    parser.add_argument("--qa-data", default=None)
    parser.add_argument("--output", default="results_perltqa.json")
    parser.add_argument("--max-characters", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in RetrievalMode],
        default=RetrievalMode.BALANCED.value,
    )
    parser.add_argument(
        "--retriever",
        choices=["memoria", "window-bm25", "window-minilm"],
        default="memoria",
    )
    args = parser.parse_args()

    if bool(args.memory_data) != bool(args.qa_data):
        parser.error("--memory-data and --qa-data must be supplied together")
    if args.memory_data:
        memory_path, qa_path = Path(args.memory_data), Path(args.qa_data)
    else:
        memory_path, qa_path = download()

    memories = json.loads(memory_path.read_text())
    qa_characters = load_character_questions(json.loads(qa_path.read_text()))
    selected = list(qa_characters.items())
    if args.max_characters is not None:
        selected = selected[:args.max_characters]

    resolved_names: dict[str, str] = {}
    records_by_character: dict[str, list[QuestionRecord]] = {}
    valid_counts = defaultdict(int)
    missing_counts = defaultdict(int)
    total_questions = 0
    for qa_name, sections in selected:
        memory_name = resolve_character_name(qa_name, sections, memories)
        resolved_names[qa_name] = memory_name
        records = list(iter_questions(qa_name, sections))
        if args.max_questions is not None:
            remaining = max(args.max_questions - total_questions, 0)
            records = records[:remaining]
        bank_ids = set(build_memory_bank(memories[memory_name]).parent_ids)
        records = [
            QuestionRecord(
                record.character,
                record.memory_type,
                record.question,
                normalize_gold_id(record.gold_id, bank_ids),
            )
            for record in records
        ]
        for record in records:
            if record.gold_id in bank_ids:
                valid_counts[record.memory_type] += 1
            else:
                missing_counts[record.memory_type] += 1
        records_by_character[qa_name] = records
        total_questions += len(records)
        if args.max_questions is not None and total_questions >= args.max_questions:
            break

    validation = {
        "characters": len(records_by_character),
        "questions": total_questions,
        "valid_gold": int(sum(valid_counts.values())),
        "missing_gold": int(sum(missing_counts.values())),
        "by_type": {
            memory_type: {
                "valid": valid_counts[memory_type],
                "missing": missing_counts[memory_type],
            }
            for memory_type in MEMORY_TYPES
        },
        "resolved_character_aliases": sum(
            qa_name != memory_name
            for qa_name, memory_name in resolved_names.items()
        ),
    }
    print(json.dumps(validation, indent=2))
    if args.validate_only:
        return

    embedder = Embedder()
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA_SQL)
    retriever = Retriever(KnowledgeGraph(db), embedder)
    mode = RetrievalMode(args.mode)
    ks = (1, 2, 3, 5)
    strict_hits = {k: [] for k in ks}
    valid_hits = {k: [] for k in ks}
    type_hits = defaultdict(lambda: {k: [] for k in ks})
    processed = 0
    started = time.time()

    for qa_name, records in records_by_character.items():
        memory_name = resolved_names[qa_name]
        bank = build_memory_bank(memories[memory_name])
        parent_set = set(bank.parent_ids)
        if args.retriever != "window-bm25":
            chunk_embeddings = retriever._emb_cache.get_batch(bank.chunk_docs)
        else:
            chunk_embeddings = None

        for record in records:
            valid = record.gold_id in parent_set
            if valid:
                if args.retriever == "memoria":
                    ranked = retriever.retrieve_grouped_sessions(
                        record.question,
                        bank.chunk_docs,
                        bank.chunk_parent_ids,
                        top_k=5,
                        mode=mode,
                    )
                    retrieved = [parent_id for parent_id, _ in ranked]
                elif args.retriever == "window-bm25":
                    retrieved = rank_window_bm25(
                        record.question,
                        bank.chunk_docs,
                        bank.chunk_parent_ids,
                        5,
                    )
                else:
                    query_embedding = embedder.embed_single(record.question)
                    scores = np.asarray(
                        chunk_embeddings @ query_embedding, dtype=float
                    )
                    retrieved = _collapse_scores(
                        scores, bank.chunk_parent_ids, 5
                    )
            else:
                retrieved = []

            for k in ks:
                hit = float(record.gold_id in retrieved[:k])
                strict_hits[k].append(hit)
                type_hits[record.memory_type][k].append(hit)
                if valid:
                    valid_hits[k].append(hit)
            processed += 1
            if processed % 500 == 0:
                elapsed = time.time() - started
                print(
                    f"  {processed}/{total_questions} "
                    f"strict-R@5={_mean(strict_hits[5]):.3f} "
                    f"({processed/elapsed:.1f} q/s)",
                    flush=True,
                )

    elapsed = time.time() - started
    strict_metrics = {f"recall@{k}": _mean(strict_hits[k]) for k in ks}
    valid_metrics = {f"recall@{k}": _mean(valid_hits[k]) for k in ks}
    by_type = {
        memory_type: {
            "total": len(type_hits[memory_type][1]),
            "metrics": {
                f"recall@{k}": _mean(type_hits[memory_type][k]) for k in ks
            },
        }
        for memory_type in MEMORY_TYPES
    }

    print("\nPerLTQA English v2 memory retrieval")
    print(f"Retriever: {args.retriever}")
    print(f"Rows: {total_questions} ({validation['missing_gold']} missing gold IDs)")
    for k in ks:
        print(
            f"R@{k}: {strict_metrics[f'recall@{k}']*100:.1f}% strict / "
            f"{valid_metrics[f'recall@{k}']*100:.1f}% valid-gold"
        )
    print(f"Time: {elapsed:.1f}s ({processed/elapsed:.1f} q/s)")

    retriever_path = Path(__file__).parent.parent / "memoria" / "retriever.py"
    output = {
        "benchmark": "PerLTQA English v2 memory retrieval",
        "upstream_commit": UPSTREAM_COMMIT,
        "dataset_sha256": {
            "memory": _sha256(memory_path),
            "qa": _sha256(qa_path),
        },
        "retriever_sha256": _sha256(retriever_path),
        "protocol": {
            "scope": "all updated English v2 questions",
            "scored_unit": "native memory record",
            "window_items": WINDOW_ITEMS,
            "window_stride": WINDOW_STRIDE,
            "metrics": [f"recall@{k}" for k in ks],
        },
        "validation": validation,
        "retriever": args.retriever,
        "mode": mode.value,
        "strict_all": {"total": total_questions, "metrics": strict_metrics},
        "valid_gold": {
            "total": len(valid_hits[1]),
            "metrics": valid_metrics,
        },
        "by_type_strict": by_type,
        "elapsed_sec": elapsed,
        "qps": processed / elapsed,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Saved to {output_path}")
    db.close()


if __name__ == "__main__":
    main()
