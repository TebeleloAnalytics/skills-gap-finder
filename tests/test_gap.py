import numpy as np
import pandas as pd

from skills_gap_finder.gap import match_skills, nominated_skills, summarise


def unit(*rows):
    v = np.asarray(rows, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# Three skills along three axes, in a 3-dimensional toy "embedding space"
SKILL_URIS = ["sql", "stats", "people"]
SKILL_VECTORS = unit([1, 0, 0], [0, 1, 0], [0, 0, 1])
REQUIRED = pd.DataFrame(
    {
        "skill_uri": SKILL_URIS,
        "label": ["SQL", "statistics", "manage staff"],
        "importance": ["essential", "essential", "optional"],
    }
)


def test_nominated_skills_takes_top_k_per_statement():
    statements = unit([1, 0.1, 0], [0, 0.2, 1])
    assert nominated_skills(statements, SKILL_VECTORS, SKILL_URIS, k=1) == [{"sql"}, {"people"}]
    assert nominated_skills(statements, SKILL_VECTORS, SKILL_URIS, k=2) == [{"sql", "stats"}, {"people", "stats"}]


def test_skill_is_covered_only_when_nominated_and_above_floor():
    statements = ["I write SQL", "I manage a team"]
    vectors = unit([1, 0.3, 0], [0, 0, 1])
    nominated = nominated_skills(vectors, SKILL_VECTORS, SKILL_URIS, k=1)

    matches = {
        m.skill_uri: m
        for m in match_skills(statements, vectors, REQUIRED, SKILL_VECTORS, nominated, min_similarity=0.5)
    }

    assert matches["sql"].covered and matches["sql"].evidence == "I write SQL"
    assert matches["people"].covered
    assert not matches["stats"].covered  # never in a statement's top 1


def test_floor_blocks_weak_matches():
    statements = ["I work hard"]
    vectors = unit([1, 1, 1])  # equally (weakly) similar to everything
    nominated = [set(SKILL_URIS)]
    matches = match_skills(statements, vectors, REQUIRED, SKILL_VECTORS, nominated, min_similarity=0.9)
    assert not any(m.covered for m in matches)


def test_summarise_counts_essential_only_and_orders_gaps():
    statements = ["I write SQL"]
    vectors = unit([1, 0.2, 0.1])
    nominated = nominated_skills(vectors, SKILL_VECTORS, SKILL_URIS, k=1)
    result = summarise(match_skills(statements, vectors, REQUIRED, SKILL_VECTORS, nominated, min_similarity=0.5))

    assert (result["essential_covered"], result["essential_total"]) == (1, 2)
    assert result["essential_coverage"] == 0.5
    assert [g["label"] for g in result["gaps"]] == ["statistics", "manage staff"]  # essential gaps first
