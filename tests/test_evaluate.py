import numpy as np
import pandas as pd
import pytest

from skills_gap_finder.evaluate import first_correct_rank, metrics, score_matrix_tfidf, threshold_curve

LABELS = np.array(["SQL", "statistics", "Python"])


def test_first_correct_rank():
    scores = np.array([0.2, 0.9, 0.5])
    assert first_correct_rank(scores, LABELS, {"statistics"}) == 1
    assert first_correct_rank(scores, LABELS, {"SQL", "Python"}) == 2  # first of several acceptable answers
    with pytest.raises(ValueError):
        first_correct_rank(scores, LABELS, {"Java"})


def test_metrics():
    result = metrics([1, 2, 10])
    assert result["hit@1"] == pytest.approx(1 / 3, abs=0.001)
    assert result["hit@5"] == pytest.approx(2 / 3, abs=0.001)
    assert result["MRR"] == pytest.approx((1 + 0.5 + 0.1) / 3, abs=0.001)


def test_tfidf_baseline_finds_exact_words():
    skills = pd.DataFrame(
        {
            "label": ["query languages", "statistics"],
            "description": ["Use SQL to retrieve data from databases.", "Collect and analyse numerical data."],
            "alt_labels": ["", ""],
        }
    )
    scores = score_matrix_tfidf(skills, ["retrieve data with SQL"])
    assert scores[0].argmax() == 0


def test_threshold_curve_is_perfect_when_scores_separate():
    scores = np.array([[0.9, 0.1, 0.1], [0.1, 0.1, 0.8]])
    curve = threshold_curve(scores, LABELS, [{"SQL"}, {"Python"}])
    best = curve.loc[curve["f1"].idxmax()]
    assert best["f1"] == 1.0
