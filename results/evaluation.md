# Retrieval evaluation

50 plain-language statements, each labelled with the correct ESCO skill(s), ranked against all 315 skills.

| method                                 |   hit@1 |   hit@5 |   MRR |
|:---------------------------------------|--------:|--------:|------:|
| TF-IDF keywords (baseline)             |    0.34 |    0.54 | 0.433 |
| sentence-transformers/all-MiniLM-L6-v2 |    0.56 |    0.84 | 0.683 |
| BAAI/bge-small-en-v1.5                 |    0.56 |    0.82 | 0.681 |
| BAAI/bge-base-en-v1.5                  |    0.66 |    0.82 | 0.751 |
| snowflake/snowflake-arctic-embed-s     |    0.34 |    0.48 | 0.418 |

Coverage rule used by gap analysis (BAAI/bge-base-en-v1.5, top 5, similarity >= 0.5): recall 82%

Per-query ranks and top results: [per_query_ranks.csv](per_query_ranks.csv)
