"""Unit tests for IR metric implementations."""

from __future__ import annotations

import math

import pytest

from src.evaluate.metrics import (
    average_precision,
    dcg_at_k,
    mean_average_precision,
    mean_metric,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k_basic():
    retrieved = ["d1", "d2", "d3", "d4"]
    qrels = {"d1": 1, "d3": 1}
    assert precision_at_k(retrieved, qrels, 2) == 0.5
    assert precision_at_k(retrieved, qrels, 4) == 0.5
    assert precision_at_k(retrieved, qrels, 0) == 0.0


def test_precision_at_k_handles_short_retrieval():
    retrieved = ["d1", "d2"]
    qrels = {"d1": 1, "d2": 1}
    # P@10 always divides by k=10 (TREC convention).
    assert precision_at_k(retrieved, qrels, 10) == 0.2


def test_recall_at_k_basic():
    retrieved = ["d1", "d2", "d3"]
    qrels = {"d1": 1, "d3": 1, "d5": 1}  # 3 relevant total
    assert recall_at_k(retrieved, qrels, 3) == pytest.approx(2 / 3)


def test_average_precision_known_example():
    # Two relevant docs at ranks 1 and 3 in a list of 5.
    retrieved = ["d1", "d2", "d3", "d4", "d5"]
    qrels = {"d1": 1, "d3": 1}
    # AP = (1/1 + 2/3) / 2 = (1.0 + 0.6667) / 2 = 0.8333
    assert average_precision(retrieved, qrels) == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


def test_average_precision_zero_when_no_relevant():
    assert average_precision(["d1", "d2"], {}) == 0.0
    assert average_precision([], {"d1": 1}) == 0.0


def test_dcg_and_ndcg_binary_perfect_ranking():
    retrieved = ["d1", "d2", "d3"]
    qrels = {"d1": 1, "d2": 1, "d3": 1}
    # DCG = 1/log2(2) + 1/log2(3) + 1/log2(4)
    expected_dcg = 1.0 / math.log2(2) + 1.0 / math.log2(3) + 1.0 / math.log2(4)
    assert dcg_at_k(retrieved, qrels, 3) == pytest.approx(expected_dcg)
    # Perfect ranking → NDCG = 1
    assert ndcg_at_k(retrieved, qrels, 3) == pytest.approx(1.0)


def test_ndcg_imperfect_ranking_strictly_less_than_one():
    qrels = {"d1": 1, "d2": 1}
    perfect = ndcg_at_k(["d1", "d2"], qrels, 2)
    swapped = ndcg_at_k(["d3", "d1", "d2"], qrels, 3)
    assert perfect == pytest.approx(1.0)
    assert 0.0 < swapped < 1.0


def test_ndcg_handles_empty_qrels_or_retrieved():
    assert ndcg_at_k([], {"d1": 1}, 5) == 0.0
    assert ndcg_at_k(["d1"], {}, 5) == 0.0


def test_ndcg_supports_graded_relevance():
    retrieved = ["d1", "d2", "d3"]
    # d2 should be ranked first ideally (grade 3 vs 1), so retrieved order is suboptimal.
    qrels = {"d1": 1, "d2": 3}
    suboptimal = ndcg_at_k(retrieved, qrels, 3)
    ideal = ndcg_at_k(["d2", "d1", "d3"], qrels, 3)
    assert ideal == pytest.approx(1.0)
    assert suboptimal < ideal


def test_mean_metric_aggregates():
    assert mean_metric([0.5, 1.0, 0.0]) == pytest.approx(0.5)
    assert mean_metric([]) == 0.0


def test_mean_average_precision_skips_queries_with_no_relevant():
    retrieved = {"q1": ["d1", "d2"], "q2": ["d3"]}
    qrels = {"q1": {"d1": 1}, "q2": {}}
    # q1 AP=1.0, q2 has no relevant → not counted.
    assert mean_average_precision(retrieved, qrels) == pytest.approx(1.0)
