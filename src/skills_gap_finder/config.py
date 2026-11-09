"""Paths and settings shared by every step of the pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
CHROMA_DIR = DATA / "chroma"
EVAL_FILE = DATA / "eval" / "queries.csv"
EMPLOYEES_FILE = DATA / "employees.json"

ESCO_API = "https://ec.europa.eu/esco/api"
ESCO_LANGUAGE = "en"

# ESCO occupations that make up the demo organisation, matched by their exact ESCO title
ROLE_TITLES = [
    "data scientist",
    "data analyst",
    "database developer",
    "software developer",
    "ICT system analyst",
    "data quality specialist",
    "business intelligence manager",
    "ICT intelligent systems designer",
    "ICT project manager",
]

COLLECTION = "esco_skills"

# A required skill counts as "covered" when it is in the top COVERAGE_TOP_K search results for one of the
# person's statements AND its similarity is at least MIN_SIMILARITY. The floor stops vague statements
# ("I work hard") from claiming skills; 93% of correct matches in the evaluation set score above it.
COVERAGE_TOP_K = 5
MIN_SIMILARITY = 0.5

LLM_MODEL = "claude-opus-5-5"
