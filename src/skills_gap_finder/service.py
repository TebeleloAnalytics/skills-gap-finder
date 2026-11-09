"""Loads the tables, embedding model and vector index once, and answers search and gap questions.

Shared by the command line (cli.py) and the API (api.py).
"""

from functools import cached_property

from . import clean, gap, index
from .config import COVERAGE_TOP_K
from .embeddings import Embedder, FastEmbedder


class SkillsService:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.occupations, self.skills, self.occupation_skills = clean.load()
        self.embed = embedder or FastEmbedder()
        self.collection = index.get_collection()
        if self.collection.count() != len(self.skills):
            raise RuntimeError("The vector index is missing or out of date. Run: uv run skills-gap build")

    @cached_property
    def _all_vectors(self):
        uris = self.skills["skill_uri"].tolist()
        return uris, index.get_vectors(self.collection, uris)

    def roles(self) -> list[str]:
        return self.occupations["role"].tolist()

    def search(self, query: str, k: int = 5) -> list[dict]:
        return index.search(self.collection, self.embed, query, k)

    def gap_analysis(self, statements: list[str], role: str) -> dict:
        statements = [s.strip() for s in statements if s and s.strip()]
        if not statements:
            raise ValueError("Give at least one statement describing your work")
        required = gap.role_skills(role, self.occupations, self.skills, self.occupation_skills)
        statement_vectors = self.embed.queries(statements)
        all_uris, all_vectors = self._all_vectors
        nominated = gap.nominated_skills(statement_vectors, all_vectors, all_uris, COVERAGE_TOP_K)
        required_vectors = index.get_vectors(self.collection, required["skill_uri"].tolist())
        matches = gap.match_skills(statements, statement_vectors, required, required_vectors, nominated)
        return {"role": role, **gap.summarise(matches)}

    def describe(self, skill_uris: list[str]) -> list[dict]:
        """ESCO label and description for each skill: the context passed to the language model."""
        rows = self.skills.set_index("skill_uri").loc[skill_uris, ["label", "description"]]
        return rows.reset_index().to_dict("records")
