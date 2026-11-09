"""Command line: uv run skills-gap <command>

ingest     download occupations and skills from the ESCO API (cached in data/raw)
clean      build and validate the tidy tables (data/processed)
build      embed the skills into the Chroma vector index (data/chroma)
evaluate   compare keyword search with embedding models, write results/
search     find the skills closest to a description
gap        skills-gap analysis for a demo employee or your own statements
plan       gap analysis plus a RAG development plan
serve      start the REST API
"""

import argparse
import json

from .config import EMPLOYEES_FILE, EVAL_FILE, ROOT


def cmd_ingest(args) -> None:
    from . import ingest

    ingest.run()


def cmd_clean(args) -> None:
    from . import clean

    clean.run()


def cmd_build(args) -> None:
    from . import clean, index
    from .embeddings import FastEmbedder

    _, skills, _ = clean.load()
    embed = FastEmbedder()
    collection = index.build(skills, embed)
    print(f"indexed {collection.count()} skills with {embed.name}")


def cmd_evaluate(args) -> None:
    import pandas as pd

    from . import clean
    from . import evaluate as ev
    from .config import COVERAGE_TOP_K, MIN_SIMILARITY
    from .embeddings import COMPARED_MODELS, DEFAULT_MODEL, FastEmbedder

    _, skills, _ = clean.load()
    methods = {"TF-IDF keywords (baseline)": lambda q: ev.score_matrix_tfidf(skills, q)}
    scores_by_method = {}
    for model in COMPARED_MODELS:
        embed = FastEmbedder(model)

        def score(queries, embed=embed, model=model):
            scores_by_method[model] = ev.score_matrix_embeddings(skills, queries, embed)
            return scores_by_method[model]

        methods[model] = score
    summary, per_query = ev.evaluate(skills, methods)

    # recall of the coverage rule used in gap analysis: correct skill in the top k AND above the floor
    eval_set = pd.read_csv(EVAL_FILE)
    labels = skills["label"].to_numpy()
    scores = scores_by_method[DEFAULT_MODEL]
    rule_hits = []
    for row, expected in zip(scores, eval_set["expected"]):
        top = row.argsort()[::-1][:COVERAGE_TOP_K]
        rule_hits.append(any(labels[i] in expected.split(";") and row[i] >= MIN_SIMILARITY for i in top))
    rule_recall = sum(rule_hits) / len(rule_hits)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    per_query.to_csv(out / "per_query_ranks.csv", index=False)
    report = [
        "# Retrieval evaluation",
        "",
        f"{len(eval_set)} plain-language statements, each labelled with the correct ESCO skill(s), ranked against all {len(skills)} skills.",
        "",
        summary.to_markdown(index=False),
        "",
        (
            f"Coverage rule used by gap analysis ({DEFAULT_MODEL}, top {COVERAGE_TOP_K}, "
            f"similarity >= {MIN_SIMILARITY}): recall {rule_recall:.0%}"
        ),
        "",
        "Per-query ranks and top results: [per_query_ranks.csv](per_query_ranks.csv)",
    ]
    (out / "evaluation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"coverage rule recall: {rule_recall:.0%}  (report written to results/evaluation.md)")


def load_people(args) -> list[dict]:
    if args.statement:
        return [{"name": "you", "target_role": args.role, "statements": args.statement}]
    people = json.loads(EMPLOYEES_FILE.read_text(encoding="utf-8"))["employees"]
    if args.employee:
        people = [p for p in people if p["id"] == args.employee]
    if args.role:
        people = [{**p, "target_role": args.role} for p in people]
    return people


def print_analysis(person: dict, result: dict, show: int = 5) -> None:
    print(
        f"\n{person['name']} -> {result['role']}: {result['essential_covered']}/{result['essential_total']} "
        f"essential skills covered ({result['essential_coverage']:.0%})"
    )
    for m in [c for c in result["covered"] if c["importance"] == "essential"]:
        print(f'  covered  {m["label"]:<45} evidence: "{m["evidence"]}"')
    for m in [g for g in result["gaps"] if g["importance"] == "essential"][:show]:
        print(f"  gap      {m['label']}")


def cmd_search(args) -> None:
    from .service import SkillsService

    for r in SkillsService().search(args.query, args.k):
        print(f"{r['similarity']:.3f}  {r['label']}  ({r['skill_type']})")


def cmd_gap(args) -> None:
    from .service import SkillsService

    service = SkillsService()
    for person in load_people(args):
        print_analysis(person, service.gap_analysis(person["statements"], person["target_role"]))


def cmd_plan(args) -> None:
    from . import rag
    from .service import SkillsService

    service = SkillsService()
    for person in load_people(args):
        analysis = service.gap_analysis(person["statements"], person["target_role"])
        print_analysis(person, analysis)
        result = rag.generate_plan(rag.build_context(analysis, service.describe))
        note = f" ({result['reason']})" if "reason" in result else ""
        print(f"\n[plan mode: {result['mode']}{note}]\n\n{result['plan']}")


def cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run("skills_gap_finder.api:app", host=args.host, port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="skills-gap", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["ingest", "clean", "build", "evaluate"]:
        sub.add_parser(name)
    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    for name in ["gap", "plan"]:
        p = sub.add_parser(name)
        p.add_argument("--employee", help="demo employee id from data/employees.json, e.g. E001")
        p.add_argument("--role", help="target role (default: the employee's target role)")
        p.add_argument("--statement", action="append", help="describe your own work (repeat for several)")
    p = sub.add_parser("serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    if args.command in ("gap", "plan") and args.statement and not args.role:
        parser.error("--statement needs --role")
    globals()[f"cmd_{args.command}"](args)
