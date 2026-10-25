"""Compare a fresh complete benchmark directory with the published artifacts.

Does not run models or modify results. Missing/partial arms are reported rather
than being silently treated as full evaluations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BENCH = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def compare_metrics(old, new):
    return {
        name: {"previous": value, "rerun": new[name], "delta": new[name] - value}
        for name, value in old.items()
    }


def quality_fields(value, prefix=""):
    """Collect all scores and denominators, excluding timing and source metadata."""
    fields = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            fields.update(quality_fields(item, path))
        elif isinstance(item, (int, float)) and (
            "@" in key
            or key
            in {
                "total",
                "source_total",
                "failures",
                "impossible",
                "skipped",
                "excluded_abstention",
                "excluded_without_user_evidence",
                "characters",
                "questions",
                "valid_gold",
                "missing_gold",
                "valid",
                "missing",
                "resolved_character_aliases",
            }
        ):
            fields[path] = item
    return fields


def qa_summary(rows):
    if not rows:
        return None

    def score(items):
        return (
            sum(bool(row["autoeval_label"]) for row in items) / len(items)
            if items
            else None
        )

    types = sorted({row["question_type"] for row in rows if not row["is_abstention"]})
    return {
        "total": len(rows),
        "unique_questions": len({row["question_id"] for row in rows}),
        "accuracy": score(rows),
        "non_abstention_accuracy": score(
            [row for row in rows if not row["is_abstention"]]
        ),
        "abstention_accuracy": score([row for row in rows if row["is_abstention"]]),
        "by_type": {
            kind: score(
                [
                    row
                    for row in rows
                    if row["question_type"] == kind and not row["is_abstention"]
                ]
            )
            for kind in types
        },
        "usage": {
            key: sum(row["usage"][key] for row in rows)
            for key in ["reader_in", "reader_out", "judge_in", "judge_out"]
        },
        "reader_models": sorted(
            {row.get("reader_model", "not_recorded") for row in rows}
        ),
        "judge_models": sorted(
            {row.get("judge_model", "not_recorded") for row in rows}
        ),
        "reader_truncations": sum(
            row.get("reader_stop_reason") == "max_tokens" for row in rows
        ),
        "judge_truncations": sum(
            row.get("judge_stop_reason") == "max_tokens" for row in rows
        ),
    }


def compare(run_dir):
    old_dir = BENCH / "results/2026-08-11"
    output = {
        "retrieval": {},
        "retrieval_audit": {},
        "controls": {},
        "controls_audit": {},
        "qa": {},
        "qa_retrieval": {},
        "missing": [],
    }
    paths = [
        (
            "longmemeval_s_balanced_faceted_freeze.json",
            "longmemeval_s_balanced.json",
            ("metrics",),
        ),
        ("longmemeval_s_flat_bm25.json", "longmemeval_s_flat_bm25.json", ("metrics",)),
        ("longmemeval_s_minilm.json", "longmemeval_s_minilm.json", ("metrics",)),
        (
            "locomo_memoria_faceted_freeze.json",
            "locomo_memoria_windowed.json",
            ("balanced", "core_four_categories", "metrics"),
        ),
        (
            "locomo_memoria_balanced.json",
            "locomo_memoria_legacy.json",
            ("balanced", "core_four_categories", "metrics"),
        ),
        (
            "locomo_flat_bm25.json",
            "locomo_flat_bm25.json",
            ("balanced", "core_four_categories", "metrics"),
        ),
        (
            "locomo_minilm.json",
            "locomo_minilm.json",
            ("balanced", "core_four_categories", "metrics"),
        ),
        (
            "perltqa_en_v2_memoria_frozen.json",
            "perltqa_en_v2_memoria.json",
            ("strict_all", "metrics"),
        ),
        (
            "perltqa_en_v2_window_bm25.json",
            "perltqa_en_v2_window_bm25.json",
            ("strict_all", "metrics"),
        ),
        (
            "perltqa_en_v2_window_minilm.json",
            "perltqa_en_v2_window_minilm.json",
            ("strict_all", "metrics"),
        ),
    ]
    for old_name, new_name, keys in paths:
        if not (run_dir / new_name).exists():
            output["missing"].append(new_name)
            continue
        old, new = read(old_dir / old_name), read(run_dir / new_name)
        old_fields, new_fields = quality_fields(old), quality_fields(new)
        output["retrieval_audit"][new_name] = {
            "all_quality_fields_identical": old_fields == new_fields,
            "fields_compared": len(old_fields),
            "differences": {
                key: {"previous": old_fields.get(key), "rerun": new_fields.get(key)}
                for key in sorted(old_fields.keys() | new_fields.keys())
                if old_fields.get(key) != new_fields.get(key)
            },
        }
        for key in keys:
            old, new = old[key], new[key]
        output["retrieval"][new_name] = compare_metrics(old, new)
    for name, section in [("controls_ce", "controls"), ("controls_ablation", "arms")]:
        path = run_dir / f"{name}.json"
        if not path.exists():
            output["missing"].append(path.name)
            continue
        old, new = read(BENCH / f"results_{name}.json"), read(path)
        if old["rows"] != new["rows"] or old["k_ce"] != new["k_ce"]:
            raise ValueError(f"{name}: changed evaluation scope or reranking budget")
        if old[section].keys() != new[section].keys():
            raise ValueError(f"{name}: missing or unexpected control arms")
        output["controls"][name] = {
            arm: compare_metrics(old[section][arm]["metrics"], values["metrics"])
            for arm, values in new[section].items()
        }
        old_fields, new_fields = (
            quality_fields(old[section]),
            quality_fields(new[section]),
        )
        output["controls_audit"][name] = {
            "all_quality_fields_identical": old_fields == new_fields,
            "fields_compared": len(old_fields),
            "differences": {
                key: {"previous": old_fields.get(key), "rerun": new_fields.get(key)}
                for key in sorted(old_fields.keys() | new_fields.keys())
                if old_fields.get(key) != new_fields.get(key)
            },
        }
    for arm in ["memoria", "hybrid-ce-dual"]:
        name = f"qa_retrieval_{arm}.json"
        if (run_dir / name).exists():
            old, new = read(BENCH / name), read(run_dir / name)
            changed = [qid for qid, ids in new.items() if old.get(qid) != ids]
            output["qa_retrieval"][arm] = {
                "total": len(new),
                "question_ids_match": old.keys() == new.keys(),
                "ordered_top5_changes": len(changed),
                "changed_ids": changed,
                "missing_question_ids": sorted(old.keys() - new.keys()),
            }
        else:
            output["missing"].append(name)
        for cot in [False, True]:
            name = f"results_qa{'_cot' if cot else ''}_{arm}.jsonl"
            path = run_dir / name
            if not path.exists():
                output["missing"].append(name)
                continue
            old = [
                json.loads(line)
                for line in (BENCH / name).read_text().splitlines()
                if line.strip()
            ]
            new = [
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            ]
            old_by_id = {row["question_id"]: row for row in old}
            new_ids = {row["question_id"] for row in new}
            unexpected_ids = new_ids - old_by_id.keys()
            if unexpected_ids:
                raise ValueError(
                    f"{name}: unexpected question IDs: {sorted(unexpected_ids)}"
                )
            new_summary = qa_summary(new)
            paired = {
                "previous_only_correct": sum(
                    bool(old_by_id[row["question_id"]]["autoeval_label"])
                    and not row["autoeval_label"]
                    for row in new
                ),
                "rerun_only_correct": sum(
                    not old_by_id[row["question_id"]]["autoeval_label"]
                    and bool(row["autoeval_label"])
                    for row in new
                ),
                "identical_reader_answers": sum(
                    old_by_id[row["question_id"]]["hypothesis"] == row["hypothesis"]
                    for row in new
                ),
                "judge_label_changes_with_identical_answer": sum(
                    old_by_id[row["question_id"]]["hypothesis"] == row["hypothesis"]
                    and old_by_id[row["question_id"]]["autoeval_label"]
                    != row["autoeval_label"]
                    for row in new
                ),
            }
            output["qa"][name] = {
                "complete": len(new) == 500 and new_ids == old_by_id.keys(),
                "missing_question_ids": sorted(old_by_id.keys() - new_ids),
                "previous": qa_summary(old),
                "rerun": new_summary,
                "paired": paired,
            }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.run_dir.resolve())
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
