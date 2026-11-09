"""Step 5: compare what a person says they do with what a role requires.

Each statement the person wrote is searched against ALL skills in the index. A required skill
is "covered" when it is among the top-k results for at least one statement and its similarity
clears a minimum floor. The statement that matched best is kept as evidence, so every result
can be explained.

Why rank and not a fixed similarity cut-off: raw similarity depends a lot on how a statement
is worded, while "is this skill one of the closest matches" is stable. On the evaluation set
the correct skill is in the top 5 for 82% of statements (hit@5), which is this rule's recall.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import MIN_SIMILARITY


@dataclass
class SkillMatch:
    skill_uri: str
    label: str
    importance: str
    similarity: float
    evidence: str  # the statement that best matches this skill
    covered: bool


def role_skills(
    role: str, occupations: pd.DataFrame, skills: pd.DataFrame, occupation_skills: pd.DataFrame
) -> pd.DataFrame:
    match = occupations[occupations["role"].str.lower() == role.lower()]
    if match.empty:
        raise KeyError(f"Unknown role {role!r}. Choose from: {', '.join(occupations['role'])}")
    links = occupation_skills[occupation_skills["occupation_uri"] == match.iloc[0]["occupation_uri"]]
    return links.merge(skills, on="skill_uri")[["skill_uri", "label", "importance", "description"]].reset_index(
        drop=True
    )


def nominated_skills(
    statement_vectors: np.ndarray, all_skill_vectors: np.ndarray, all_skill_uris: list[str], k: int
) -> list[set[str]]:
    """For each statement, the URIs of its k most similar skills across the whole index."""
    similarity = statement_vectors @ all_skill_vectors.T
    top = np.argsort(-similarity, axis=1)[:, :k]
    return [{all_skill_uris[i] for i in row} for row in top]


def match_skills(
    statements: list[str],
    statement_vectors: np.ndarray,
    required: pd.DataFrame,
    required_vectors: np.ndarray,
    nominated: list[set[str]],
    min_similarity: float = MIN_SIMILARITY,
) -> list[SkillMatch]:
    """Decide coverage for each required skill. Vectors must be L2-normalised, one row per item."""
    similarity = required_vectors @ statement_vectors.T  # rows: required skills, columns: statements
    matches = []
    for i, row in enumerate(required.itertuples()):
        # statements that rank this skill in their top k and clear the floor
        supporting = [
            j for j, uris in enumerate(nominated) if row.skill_uri in uris and similarity[i, j] >= min_similarity
        ]
        best = max(supporting, key=lambda j: similarity[i, j]) if supporting else int(similarity[i].argmax())
        matches.append(
            SkillMatch(
                skill_uri=row.skill_uri,
                label=row.label,
                importance=row.importance,
                similarity=round(float(similarity[i, best]), 3),
                evidence=statements[best],
                covered=bool(supporting),
            )
        )
    return matches


def summarise(matches: list[SkillMatch]) -> dict:
    essential = [m for m in matches if m.importance == "essential"]
    covered = [m for m in essential if m.covered]
    # closest gaps first: they're the easiest to close
    gaps = sorted((m for m in matches if not m.covered), key=lambda m: (m.importance != "essential", -m.similarity))
    return {
        "essential_coverage": round(len(covered) / len(essential), 3) if essential else None,
        "essential_covered": len(covered),
        "essential_total": len(essential),
        "covered": [asdict(m) for m in sorted(matches, key=lambda m: -m.similarity) if m.covered],
        "gaps": [asdict(m) for m in gaps],
    }
