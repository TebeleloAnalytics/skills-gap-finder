from skills_gap_finder.clean import build_tables, tidy_text, validate


def skill(uri, title, desc="A description.", alt=None, kind="skill"):
    record = {
        "uri": uri,
        "title": title,
        "description": {"en": {"literal": desc}},
        "_links": {"hasSkillType": [{"title": kind}]},
    }
    if alt:
        record["alternativeLabel"] = {"en": alt}
    return record


def occupation(uri, essential, optional=()):
    return {
        "uri": uri,
        "description": {"en": {"literal": "Does things."}},
        "_links": {
            "hasEssentialSkill": [{"uri": u} for u in essential],
            "hasOptionalSkill": [{"uri": u} for u in optional],
        },
    }


def test_tidy_text_collapses_whitespace():
    assert tidy_text("  perform\n data   cleansing ") == "perform data cleansing"
    assert tidy_text(None) == ""


def test_build_tables_shapes_and_duplicate_links():
    occupations_raw = {"analyst": occupation("o1", essential=["s1", "s2"], optional=["s2"])}
    skills_raw = {"s1": skill("s1", "statistics", alt=["stats", "stats"]), "s2": skill("s2", "SQL", kind="knowledge")}
    occupations, skills, links = build_tables(occupations_raw, skills_raw)

    assert list(occupations["role"]) == ["analyst"]
    assert skills.set_index("skill_uri").loc["s1", "alt_labels"] == "stats"  # duplicates removed
    assert skills.set_index("skill_uri").loc["s2", "skill_type"] == "knowledge"
    # s2 is listed as essential and optional: it's kept once, as essential
    assert len(links) == 2
    assert set(links["importance"]) == {"essential"}
    assert validate(occupations, skills, links) == []


def test_validate_flags_problems():
    occupations_raw = {"analyst": occupation("o1", essential=["s1", "missing"])}
    skills_raw = {"s1": skill("s1", "statistics", desc="")}
    problems = validate(*build_tables(occupations_raw, skills_raw))

    assert any("empty description" in p for p in problems)
    assert any("weren't downloaded" in p for p in problems)
