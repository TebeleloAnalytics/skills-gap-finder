"""Step 4: measure retrieval accuracy on a hand-labelled set of 50 plain-language statements.

Each statement ("I build Power BI dashboards for executives") has one or more correct ESCO
skills. Every method ranks all skills for every statement and is scored on:
  hit@1  the top result is a correct skill
  hit@5  a correct skill is in the top 5
  MRR    mean reciprocal rank: 1 / position of the first correct skill, averaged

The keyword baseline (TF-IDF) shows how much the embeddings actually add.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import EVAL_FILE
from .embeddings import Embedder, skill_document


def first_correct_rank(scores: np.ndarray, labels: np.ndarray, expected: set[str]) -> int:
    """1-based position of the first correct skill when skills are sorted by score, highest first."""
    order = np.argsort(-scores, kind="stable")
    for position, i in enumerate(order, 1):
        if labels[i] in expected:
            return position
    raise ValueError(f"None of {expected} are in the skill list")


def metrics(ranks: list[int]) -> dict:
    ranks = np.asarray(ranks)
    return {
        "hit@1": round(float((ranks == 1).mean()), 3),
        "hit@5": round(float((ranks <= 5).mean()), 3),
        "MRR": round(float((1 / ranks).mean()), 3),
    }


def score_matrix_embeddings(skills: pd.DataFrame, queries: list[str], embed: Embedder) -> np.ndarray:
    """Cosine similarity of every query (rows) to every skill (columns)."""
    skill_vectors = embed.documents(skills.apply(skill_document, axis=1).tolist())
    return embed.queries(queries) @ skill_vectors.T


def score_matrix_tfidf(skills: pd.DataFrame, queries: list[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    skill_matrix = vectorizer.fit_transform(skills.apply(skill_document, axis=1))
    return (vectorizer.transform(queries) @ skill_matrix.T).toarray()


def evaluate(skills: pd.DataFrame, methods: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score each method. `methods` maps a name to a function(queries) -> score matrix.

    Returns a summary table (one row per method) and a per-query table of ranks.
    """
    eval_set = pd.read_csv(EVAL_FILE)
    queries = eval_set["query"].tolist()
    expected = [set(e.split(";")) for e in eval_set["expected"]]
    labels = skills["label"].to_numpy()

    per_query = eval_set.copy()
    summary = []
    for name, score in methods.items():
        scores = score(queries)
        ranks = [first_correct_rank(row, labels, exp) for row, exp in zip(scores, expected)]
        per_query[f"rank_{name}"] = ranks
        per_query[f"top1_{name}"] = [labels[np.argmax(row)] for row in scores]
        summary.append({"method": name, **metrics(ranks)})
    return pd.DataFrame(summary), per_query


def threshold_curve(scores: np.ndarray, labels: np.ndarray, expected: list[set[str]]) -> pd.DataFrame:
    """Precision, recall and F1 of "similarity >= t means it's this skill", over all statement-skill pairs."""
    positive = np.array([[label in exp for label in labels] for exp in expected])
    rows = []
    for t in np.arange(0.20, 0.80, 0.025):
        predicted = scores >= t
        tp = (predicted & positive).sum()
        precision = tp / predicted.sum() if predicted.sum() else 1.0
        recall = tp / positive.sum()
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"threshold": round(float(t), 3), "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows).round(3)
