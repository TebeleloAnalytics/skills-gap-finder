"""Step 1: extract occupations and their skills from the ESCO REST API.

Raw API responses are cached as JSON in data/raw/, so the rest of the pipeline
runs offline and the API is only called for records we don't have yet.
"""

import json
import time

import requests

from .config import ESCO_API, ESCO_LANGUAGE, RAW, ROLE_TITLES

session = requests.Session()
session.headers["User-Agent"] = "skills-gap-finder (portfolio project)"


def get_json(path: str, params: dict, retries: int = 3) -> dict:
    """GET an ESCO endpoint, retrying with a growing pause on network errors or 5xx responses."""
    for attempt in range(1, retries + 1):
        try:
            response = session.get(f"{ESCO_API}/{path}", params=params, timeout=30)
            if response.status_code < 500:
                response.raise_for_status()
                return response.json()
        except (requests.ConnectionError, requests.Timeout):
            if attempt == retries:
                raise
        time.sleep(2 * attempt)
    response.raise_for_status()
    return response.json()


def find_occupation(title: str) -> dict:
    """Return the occupation whose ESCO title matches exactly (search results are ranked, not exact)."""
    data = get_json("search", {"text": title, "type": "occupation", "language": ESCO_LANGUAGE, "limit": 10})
    for result in data["_embedded"]["results"]:
        if result["title"].lower() == title.lower():
            return get_json("resource/occupation", {"uri": result["uri"], "language": ESCO_LANGUAGE})
    raise LookupError(f"No ESCO occupation titled {title!r}")


def load_cache(name: str) -> dict:
    path = RAW / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_cache(name: str, records: dict) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"{name}.json").write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")


def run(pause: float = 0.1) -> None:
    occupations = load_cache("occupations")
    for title in ROLE_TITLES:
        if title not in occupations:
            occupations[title] = find_occupation(title)
            print(f"occupation: {title}")
    save_cache("occupations", occupations)

    skill_uris = {
        link["uri"]
        for occupation in occupations.values()
        for relation in ("hasEssentialSkill", "hasOptionalSkill")
        for link in occupation["_links"].get(relation, [])
    }
    skills = load_cache("skills")
    missing = sorted(skill_uris - skills.keys())
    print(f"{len(skill_uris)} skills linked, {len(missing)} to download")
    for i, uri in enumerate(missing, 1):
        skills[uri] = get_json("resource/skill", {"uri": uri, "language": ESCO_LANGUAGE})
        if i % 50 == 0:
            save_cache("skills", skills)  # checkpoint, so an interrupted run resumes where it stopped
            print(f"  {i}/{len(missing)}")
        time.sleep(pause)
    save_cache("skills", skills)
    print(f"done: {len(occupations)} occupations, {len(skills)} skills cached in {RAW}")
