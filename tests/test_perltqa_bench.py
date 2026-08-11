"""Schema and rendering tests for the untouched PerLTQA holdout adapter."""

import pytest

from benchmarks.perltqa_bench import (
    _overlapping_windows,
    build_memory_bank,
    iter_questions,
    normalize_gold_id,
    parse_reference_ids,
    resolve_character_name,
)


def _memory(prefix="7"):
    return {
        "profile": {
            "Protagonist": "Ada",
            "Awards and Role Models": "A research medal and Grace Hopper",
        },
        "profile_description": "Ada is a systems researcher.",
        "social_relationship": {
            f"{prefix}_0": {
                "Supporting Characters": "Bruno",
                "Description": "They collaborate on storage systems.",
                "Relationship": "colleague",
            }
        },
        "events": {
            f"{prefix}_0_0": {
                "content": "One. Two. Three. Four. Five.",
                "summary": "Ada completed a storage project.",
                "Characters": ["Ada", "Bruno"],
                "Creation Time": "2025-01-01",
                "Last Accessed Time": "2025-02-01",
                "Theme": "research",
            }
        },
        "dialogues": {
            f"{prefix}_0_0#0": {
                "events": f"{prefix}_0_0",
                "contents": {
                    "2025-01-01": [
                        "Ada: turn one",
                        "Bruno: turn two",
                        "Ada: turn three",
                        "Bruno: turn four",
                        "Ada: turn five",
                    ]
                },
            }
        },
    }


def _questions(prefix="7"):
    row = lambda question, reference: {
        "Question": question,
        "Answer": "unused by retrieval",
        "Reference Memory": reference,
        "Memory Anchors": [],
    }
    return {
        "profile": [row("Who inspired Ada?", "Role Models")],
        "social_relationship": [
            {f"{prefix}_0": [row("Who works with Ada?", f"['{prefix}_0']")]}
        ],
        "events": [
            {f"{prefix}_0_0": [row("What project finished?", f"['{prefix}_0_0']")]}
        ],
        "dialogues": [
            {
                f"{prefix}_0_0#0": [
                    row("What did they discuss?", f"['{prefix}_0_0#0']")
                ]
            }
        ],
    }


def test_reference_parser_normalizes_profile_aliases_and_list_strings():
    assert parse_reference_ids("profile", "Role Models") == ["profile:Role Models"]
    assert normalize_gold_id(
        "profile:Role Models", {"profile:Awards and Role Models"}
    ) == "profile:Awards and Role Models"
    assert normalize_gold_id(
        "profile:Role Models", {"profile:Role Models", "profile:Awards and Role Models"}
    ) == "profile:Role Models"
    assert parse_reference_ids("events", "['7_0_0']") == ["events:7_0_0"]
    with pytest.raises(ValueError, match="Malformed"):
        parse_reference_ids("events", "not a list")


def test_character_alias_resolution_uses_unique_memory_id_prefix():
    assert resolve_character_name(
        "translated alias", _questions(), {"Ada": _memory(), "Other": _memory("9")}
    ) == "Ada"


def test_memory_bank_preserves_native_parent_ids_and_windows_long_records():
    bank = build_memory_bank(_memory())

    assert "profile:Awards and Role Models" in bank.parent_ids
    assert "events:7_0_0" in bank.parent_ids
    assert "dialogues:7_0_0#0" in bank.parent_ids
    assert bank.chunk_parent_ids.count("events:7_0_0") == 2
    assert bank.chunk_parent_ids.count("dialogues:7_0_0#0") == 2
    assert len(bank.chunk_docs) == len(bank.chunk_parent_ids)


def test_unindexed_relationship_narrative_remains_searchable_but_not_gold():
    memory = _memory()
    memory["social_relationship"] = "Ada and Bruno are colleagues."
    bank = build_memory_bank(memory)

    assert "social_relationship:unstructured" in bank.parent_ids
    assert "social_relationship:7_0" not in bank.parent_ids


def test_question_flattening_keeps_one_gold_native_record_per_row():
    records = list(iter_questions("Ada", _questions()))
    assert len(records) == 4
    assert {record.memory_type for record in records} == {
        "profile", "social_relationship", "events", "dialogues"
    }
    assert all(record.gold_id for record in records)


def test_window_builder_includes_tail_without_duplicate_windows():
    assert _overlapping_windows(["1", "2", "3", "4", "5"], 4, 2) == [
        ["1", "2", "3", "4"],
        ["2", "3", "4", "5"],
    ]
