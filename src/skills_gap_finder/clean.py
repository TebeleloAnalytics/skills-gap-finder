"""Step 2: turn raw ESCO JSON into three tidy tables and validate them.

occupations.csv        one row per role
skills.csv             one row per skill (label, type, description, alternative labels)
occupation_skills.csv  which skills each role needs, and whether essential or optional
"""

import json

import pandas as pd

from .config import PROCESSED, RAW


def tidy_text(value: str | None) -> str:
    """Collapse runs of whitespace and line breaks into single spaces."""
    return " ".join((value or "").split())


def english(field: dict | None) -> str:
    return tidy_text(((field or {}).get("en") or {}).get("literal"))


def build_tables(occupations_raw: dict, skills_raw: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    occupations = pd.DataFrame(
        {
            "occupation_uri": o["uri"],
            "role": title,
            "description": english(o.get("description")),
        }
        for title, o in occupations_raw.items()
    )

    skills = pd.DataFrame(
        {
            "skill_uri": uri,
            "label": tidy_text(s.get("title")),
            "skill_type": (s["_links"].get("hasSkillType") or [{}])[0].get("title", ""),
            "description": english(s.get("description")),
            "alt_labels": "; ".join(sorted({tidy_text(a) for a in (s.get("alternativeLabel") or {}).get("en", [])})),
        }
        for uri, s in skills_raw.items()
    )

    occupation_skills = pd.DataFrame(
        {
            "occupation_uri": o["uri"],
            "skill_uri": link["uri"],
            "importance": "essential" if relation == "hasEssentialSkill" else "optional",
        }
        for o in occupations_raw.values()
        for relation in ("hasEssentialSkill", "hasOptionalSkill")
        for link in o["_links"].get(relation, [])
    )
    # A skill listed as both essential and optional for one role counts as essential
    occupation_skills = (
        occupation_skills.sort_values("importance")  # "essential" sorts before "optional"
        .drop_duplicates(["occupation_uri", "skill_uri"])
        .reset_index(drop=True)
    )
    return occupations, skills, occupation_skills


def validate(occupations: pd.DataFrame, skills: pd.DataFrame, occupation_skills: pd.DataFrame) -> list[str]:
    """Return a list of data quality problems. An empty list means the data passed every check."""
    problems = []
    for name, df, key in [("occupations", occupations, "occupation_uri"), ("skills", skills, "skill_uri")]:
        if df[key].duplicated().any():
            problems.append(f"{name}: {df[key].duplicated().sum()} duplicate {key} values")
    for column in ["label", "description"]:
        blank = (skills[column].str.len() == 0).sum()
        if blank:
            problems.append(f"skills: {blank} rows with an empty {column}")
    orphans = ~occupation_skills["skill_uri"].isin(skills["skill_uri"])
    if orphans.any():
        problems.append(f"occupation_skills: {orphans.sum()} links to skills that weren't downloaded")
    unknown_roles = ~occupation_skills["occupation_uri"].isin(occupations["occupation_uri"])
    if unknown_roles.any():
        problems.append(f"occupation_skills: {unknown_roles.sum()} links to unknown occupations")
    roles_without_essentials = set(occupations["occupation_uri"]) - set(
        occupation_skills.loc[occupation_skills["importance"] == "essential", "occupation_uri"]
    )
    if roles_without_essentials:
        problems.append(f"occupations: {len(roles_without_essentials)} roles have no essential skills")
    return problems


def run() -> None:
    occupations_raw = json.loads((RAW / "occupations.json").read_text(encoding="utf-8"))
    skills_raw = json.loads((RAW / "skills.json").read_text(encoding="utf-8"))
    tables = build_tables(occupations_raw, skills_raw)

    problems = validate(*tables)
    if problems:
        raise SystemExit("Data quality checks failed:\n  " + "\n  ".join(problems))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, df in zip(["occupations", "skills", "occupation_skills"], tables):
        df.to_csv(PROCESSED / f"{name}.csv", index=False)
        print(f"{name}.csv: {len(df)} rows")
    print("all data quality checks passed")


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the processed tables (run `clean` first)."""
    names = ["occupations", "skills", "occupation_skills"]
    return tuple(pd.read_csv(PROCESSED / f"{n}.csv", keep_default_na=False) for n in names)
