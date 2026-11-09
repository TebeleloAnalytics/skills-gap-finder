# Skills Gap Finder

**Find the skills gap between what someone does today and what a role needs, using semantic search over the EU's ESCO skills classification. Every match is explained and the search accuracy is measured.**

A person describes their work in plain sentences ("I build Power BI dashboards for the executive team"). The system:

1. matches each sentence to standard skills with **embeddings** stored in a **vector database** (Chroma),
2. compares those skills with the skills a target role requires and reports **coverage, evidence and gaps**,
3. optionally writes a **development plan with retrieval-augmented generation (RAG)**: Claude only sees the official descriptions of the retrieved gap skills,
4. serves it all through a **REST API** (FastAPI).

Retrieval quality is **measured, not assumed**: a hand-labelled evaluation set compares a keyword baseline with four embedding models.

---

## Results

Retrieval accuracy on 50 hand-labelled, plain-language statements, each ranked against all 315 skills
(`uv run skills-gap evaluate`, full report in [results/evaluation.md](results/evaluation.md)):

| Method | hit@1 | hit@5 | MRR |
|---|---|---|---|
| TF-IDF keywords (baseline) | 0.34 | 0.54 | 0.433 |
| all-MiniLM-L6-v2 | 0.56 | **0.84** | 0.683 |
| bge-small-en-v1.5 | 0.56 | 0.82 | 0.681 |
| **bge-base-en-v1.5 (used)** | **0.66** | 0.82 | **0.751** |
| snowflake-arctic-embed-s | 0.34 | 0.48 | 0.418 |

- Embeddings find the right skill first **nearly twice as often** as keyword search (66% vs 34%), because people rarely use ESCO's wording.
- `bge-base` has the best top-1 accuracy and MRR, so it's the default. MiniLM is a reasonable lighter option (about 80 MB vs 440 MB).
- `arctic-embed-s` did no better than keywords on these short statements. I didn't investigate further, but it's a reminder to test a model on your own data rather than trust its reputation.
- The coverage rule used for gap analysis (top 5 and similarity ≥ 0.5) finds the correct skill for **82%** of statements.

## How it works

```
ESCO REST API ──► ingest ──► clean + validate ──► embed (BGE, local ONNX) ──► Chroma vector index
  (JSON)          raw cache     3 tidy tables      768-dim vectors              315 skills
                                                                                   │
"I clean messy spreadsheets..." ──► embed ──► top-5 search per statement ──────────┤
                                                                                   ▼
                                           gap analysis: covered / gaps + evidence
                                                                                   │
                                   RAG: gap skills + ESCO descriptions ──► Claude ──► development plan
```

| Step | Module | What it does |
|---|---|---|
| Ingest | `ingest.py` | Calls the ESCO API for 9 data and ICT occupations and all 315 linked skills, with retries, polite pacing and a resumable JSON cache |
| Clean | `clean.py` | Builds `occupations`, `skills` and `occupation_skills` tables. Validation checks for duplicate keys, empty labels or descriptions, broken links, and roles without essential skills. The pipeline stops if any check fails |
| Index | `index.py`, `embeddings.py` | Embeds each skill's name, description and synonyms with `BAAI/bge-base-en-v1.5` (local, CPU, no API key) into a persistent Chroma collection |
| Evaluate | `evaluate.py` | Ranks all skills for 50 labelled statements. Reports hit@1, hit@5 and MRR for a TF-IDF baseline and 4 embedding models |
| Gap analysis | `gap.py`, `service.py` | A required skill is **covered** if it's in the top 5 results for one of the person's statements and its similarity is at least 0.5. The matching statement is returned as evidence |
| RAG plan | `rag.py` | Retrieves the closest essential gaps plus their ESCO descriptions, and asks Claude for a plan grounded only in that context. Falls back to an offline plan without credentials |
| API | `api.py` | `GET /roles`, `GET /search`, `POST /gap`, `POST /plan`, with input validation |

### Design decisions

- **Why a rank rule and not a similarity cut-off for "covered"?** Raw similarity depends heavily on wording. With BGE, even unrelated pairs score about 0.45, and the correct skill for a short sentence can score only 0.55. A fixed cut-off chosen for best F1 on the evaluation set (0.65) marked a data analyst as having **0 of 23** data-scientist skills, which is obviously wrong. "Is this skill one of the 5 closest matches?" is much more stable, and its recall is exactly what hit@5 measures.
- **Why a 0.5 floor?** Without it, a vague sentence like "I work hard" would still "cover" its 5 nearest skills. 93% of correct matches in the evaluation set score above 0.5.
- **Why embed synonyms and descriptions, not just names?** ESCO names are formal ("utilise regular expressions"), while people write "I use regex". The alternative labels and descriptions give the model the vocabulary to connect them.
- **Why local embeddings?** No cost, no API key, and no personal text leaves the machine. The LLM is only used in the optional plan step.
- **Strict scoring.** The evaluation labels were written *before* seeing any results and were never changed afterwards, even where a "miss" is arguably right (e.g. the model returns the more specific *SPARQL* skill for a SPARQL question).

## Quick start

Requires [uv](https://docs.astral.sh/uv/). The cleaned ESCO tables are already in `data/processed/`, so nothing needs to be downloaded from ESCO.

```bash
uv sync
uv run skills-gap build                 # embed the 315 skills (first run downloads the ~440 MB model)
uv run skills-gap search "I use regex to extract house numbers from addresses"
uv run skills-gap gap --employee E001   # demo person: data analyst -> data scientist
uv run skills-gap gap --role "data analyst" --statement "I write SQL to pull sales data" --statement "I build Power BI dashboards"
uv run skills-gap plan --employee E001  # RAG plan: uses Claude if credentials are set, otherwise offline
uv run skills-gap serve                 # API docs at http://127.0.0.1:8000/docs
```

Other commands: `ingest` (refresh from the ESCO API), `clean` (rebuild and validate the tables), `evaluate` (re-run the comparison and write `results/`).

To enable the LLM step, set `ANTHROPIC_API_KEY`. The plan uses `claude-opus-5-5` with server-side fallbacks enabled, so if the model declines a request, the API retries it on a suitable fallback model.

```bash
uv run pytest      # 21 tests, run in seconds with toy vectors (no model download)
uv run ruff check .
```

## Example

```
$ uv run skills-gap gap --employee E001
Lerato M. -> data scientist: 7/23 essential skills covered (30%)
  covered  perform data cleansing                        evidence: "I clean messy spreadsheets: removing duplicates and fixing missing values"
  covered  normalise data                                evidence: "I clean messy spreadsheets: removing duplicates and fixing missing values"
  covered  deliver visual presentation of data           evidence: "I present monthly findings to managers"
  covered  implement data quality processes              evidence: "I clean messy spreadsheets: removing duplicates and fixing missing values"
  covered  establish data processes                      evidence: "I use Excel pivot tables and formulas every day"
  covered  execute analytical mathematical calculations  evidence: "I use Excel pivot tables and formulas every day"
  covered  online analytical processing                  evidence: "I use Excel pivot tables and formulas every day"
  gap      query languages
  gap      resource description framework query language
  gap      design database scheme
  gap      develop data processing applications
  gap      interpret current data
```

## Limitations and next steps

- **Short, specific statements beat long ones.** "I write SQL queries" matches specific query languages (LINQ, XQuery) more closely than ESCO's general "query languages" skill. Splitting skills into specific versus general groups, or re-ranking with a cross-encoder, would help.
- **ESCO is European** and broad. A South African deployment would add local occupations, SAQA/NQF qualifications and company-specific skills.
- **The evaluation set is small (50 statements)**, written by one person. More statements from real (anonymised) job descriptions would make the numbers more reliable.
- **Self-description isn't proof.** In a real platform, statements would come with evidence: project outputs, manager validation, assessments.
- Next: a feedback loop where people confirm or reject matches (which builds more evaluation data), and tracking coverage over time per team.

## Data and privacy

- Skills data: [ESCO](https://esco.ec.europa.eu) (European Skills, Competences, Qualifications and Occupations), © European Union, used under the European Commission's reuse policy.
- The people in `data/employees.json` are **fictional**. No real personal data is used anywhere. Employee statements are embedded locally. Only the plan step sends text to an external API, and only the role, the gap skills and the matched statements.

## How this was built

This project was built with an AI coding assistant (Claude Code), which wrote most of the code under my direction. I reviewed the design decisions, the evaluation and the results, and I can explain how each part works and why. The project history is honest about what didn't work: the first similarity threshold gave obviously wrong answers and was replaced by the rank-based rule described above.

---

Built by [Tebelelo Selowa](https://github.com/TebeleloAnalytics) · MIT licence
