import math
import sys
from pathlib import Path

import pytest

import benchmarks.locomo_bench as locomo_module
import benchmarks.longmemeval_final as longmemeval_module

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))

from retrieval_metrics import (  # noqa: E402
    ndcg_any_at_k,
    rank_dense,
    rank_flat_bm25,
    recall_all_at_k,
    recall_any_at_k,
)
from locomo_bench import build_session_windows  # noqa: E402


def test_recall_any_and_all_are_not_conflated():
    retrieved = ["a", "x", "y"]
    correct = {"a", "b"}

    assert recall_any_at_k(retrieved, correct, 3) == 1.0
    assert recall_all_at_k(retrieved, correct, 3) == 0.0


def test_empty_ground_truth_never_scores_as_recalled():
    assert recall_any_at_k(["a"], set(), 1) == 0.0
    assert recall_all_at_k(["a"], set(), 1) == 0.0
    assert ndcg_any_at_k(["a"], set(), 1) == 0.0


def test_ndcg_uses_all_gold_items_in_ideal_ranking():
    # Retrieving one of two gold items is not a perfect ranking.
    score = ndcg_any_at_k(["a", "x"], {"a", "b"}, 2)
    assert score == pytest.approx(0.5)


def test_ndcg_matches_longmemeval_discount_convention():
    # The official implementation leaves ranks one and two undiscounted.
    expected = (1.0 + 1.0 / math.log2(3)) / 2.0
    assert ndcg_any_at_k(["x", "a", "b"], {"a", "b"}, 3) == pytest.approx(expected)


def test_flat_bm25_and_dense_baselines_return_ranked_ids():
    ids = ["irrelevant", "relevant"]
    docs = ["blue ocean", "database postgres database"]
    assert rank_flat_bm25("postgres", docs, ids, 1) == ["relevant"]

    import numpy as np

    document_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert rank_dense(
        np.array([0.0, 1.0]), document_embeddings, ids, 1
    ) == ["relevant"]


def test_locomo_windows_include_dates_captions_and_parent_ids():
    conversation = {
        "session_1": [
            {"speaker": "A", "text": "one"},
            {"speaker": "B", "text": "two", "blip_caption": "a red bicycle"},
            {"speaker": "A", "text": "three"},
        ],
        "session_1_date_time": "1 pm on 2 May, 2025",
    }

    chunks, parent_ids = build_session_windows(
        conversation, window_turns=2, stride=1
    )

    assert chunks
    assert parent_ids == ["session_1"] * len(chunks)
    assert all("Session date: 1 pm on 2 May, 2025" in chunk for chunk in chunks)
    assert any("Image caption: a red bicycle" in chunk for chunk in chunks)


def test_benchmark_runners_are_importable_modules():
    assert callable(locomo_module.build_session_windows)
    assert callable(longmemeval_module.build_session_documents)
