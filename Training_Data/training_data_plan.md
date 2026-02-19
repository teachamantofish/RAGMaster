# Training Data Directory Plan

## Purpose

Create `Training_Data/` as a shared training data preparation directory. The synthetic query generator lives here, producing queries that feed the reranker training pipeline. The embed model's triplet generation stays independent (it generates triplets from chunk relationships, not queries).

## Directory Structure

```
Training_Data/
  config_training_data.py          # HF API model, paths, prompts, target counts
  generate_synthetic_queries.py    # Stage 1: read chunks, call HF API, produce candidate queries
  judge_relevance.py               # Stage 2: retrieve top-K, LLM judges multi-chunk positive_ids
  retriever_eval_queries.json      # MOVED here from Train_Reranker_Model/ (canonical location)
```

Both `Train_Embed_Model/` and `Train_Reranker_Model/` will reference `Training_Data/retriever_eval_queries.json` as the single source of truth.

## What to KEEP in Data/framemaker/mif_jsx/

| Item | Verdict | Reason |
|------|---------|--------|
| `a_chunks.json` | KEEP | Master chunk file — content is fixed |
| `Qwen3-Embedding-0.6B/` | KEEP | Base embedding model |
| `embed_model_adapter/` | KEEP for now | Current fine-tuned embed model (will be retrained) |
| `embedding_training_data/` | KEEP | Original embed triplets (epoch 1 data) |
| `embedding_training_data1/` | KEEP | Retrained embed triplets (current) |
| `master_rerank_training_log.csv` | KEEP | Historical experiment log |

## What to DELETE in Data/framemaker/mif_jsx/

| Item | Reason |
|------|--------|
| `rerank_model_adapter/` | Exp 38 DeBERTa checkpoint — worse than MiniLM, will retrain |
| `a_embeddings.parquet` | Will be regenerated after embed retrain |
| `a_chunks_postembedding.json` | Will be regenerated |
| `reranker_training_data/` | Derived cross-encoder pairs — will be regenerated from expanded queries |
| `Qwen3-Reranker-0.6B/` | Downloaded but never used (LLM-based rerankers rejected) |
| `a_*.log` files | Transient log artifacts |
| `.retrieval_stats_*.json` | Transient artifacts |

## Execution Order

1. DONE **Re-run summarization** (`AI_RAG/pipeline/4.01summary.py`)
   - Regenerates `chunk_summary` and `page_summary` from the now-corrected `content` fields
   - Old summaries were based on content with 500 broken API names

2. DONE **Generate synthetic queries** (`Training_Data/generate_synthetic_queries.py`)
   - Read chunks from `a_chunks.json`
   - Call HF Inference API (qwen3-next-80b-a3b-instruct-hmt) to produce ~500 candidate queries
   - Source chunk becomes initial candidate positive_id
   - Tag with `"source": "synthetic"` to distinguish from hand-labeled

3. DONE **Call HF Inference API** (qwen3-next-80b-a3b-instruct-hmt) for relevance judging
   - Per-query scoring used during judging phase

4. DONE **Judge relevance** (`Training_Data/judge_relevance.py`)
   - For each generated query, retrieve top-K candidates via existing retrieval (BM25 + dense)
   - Send query + K candidates to LLM: "Which chunks answer this query? Rate 0-2."
   - Assign proper multi-chunk `positive_ids` (not just the source chunk)

   **Required Process (cost-safe):**
   1. Run `Training_Data/judge_relevance.py`.
   2. Complete **100% local preprocessing first** (load queries/chunks, build BM25 index, prepare dense retrieval state).
   3. Only after local prep is complete, resume HF endpoint and begin LLM judging calls.
   4. Print visible progress continuously, including:
      - local stage transitions (`[DATA]`, `[RETRIEVAL]`)
      - endpoint lifecycle (`[HF] endpoint resumed`, `[HF] endpoint paused`)
      - **per-query** call status (`[HF] Judging query X/Y (qid)`)
   5. On normal exit or interruption, auto-pause HF endpoint to stop billing.

   **Run command (max quality):**
   - `cd C:\GIT\Z_Master_Rag\Training_Data`
   - `$env:PYTHONIOENCODING='utf-8'; python -u .\judge_relevance.py --resume`

5. DONE **Merge into retriever_eval_queries.json**
   - Preserve existing 216 hand-labeled queries
   - Append synthetic queries with proper IDs (q_synth_001, q_synth_002, ...)

6. DONE **Retrain embedding model** (`Train_Embed_Model/`)
   - More triplets from expanded query set
   - Re-embed all chunks → update vector store

7. DONE **Retrain reranker** (`Train_Reranker_Model/`)
   - Train with `microsoft/deberta-v3-base` using the expanded synthetic query set
   - More cross-encoder pairs from expanded queries
   - Compare against Exp 37 baseline (MRR 0.811)
   - Note: run used `--data-allow-missed-positives` to proceed past 8 queries with no positives in top-200.

## Path Reference Updates

- `Train_Reranker_Model/scripts/config_training_rerank.py` line ~90:
  `queries_file` → `Training_Data/retriever_eval_queries.json`
- Embed model doesn't currently use queries (chunk-relationship triplets), update later if needed

## Verification Checkpoints

- After summarization: spot-check 5-10 chunk_summaries that had broken code names
- After query generation: manual review of 20-30 synthetic queries for quality
- After relevance judging: verify synthetic queries have 1-3 positive_ids each
- After reranker retrain: compare MRR against Exp 37 baseline (0.811)

## Design Decisions

- Embed model triplet generation stays independent (chunk-relationship-based, not query-based)
- `retriever_eval_queries.json` moves to `Training_Data/` as canonical source
- Existing 216 hand-labeled queries preserved; synthetic queries appended
- Use DeBERTa-v3-base for reranker retraining after data expansion (previous underperformance was with limited query volume)

## Human Holdout Rationale (Why + When)

The human-written query set is treated as a **quality gate**, not as training data.

- **Why eval-only:**
   - Human queries better represent real user phrasing and intent.
   - Keeping them out of training prevents contamination and inflated metrics.
   - They reduce synthetic leakage risk (token-copy, heading-copy, filter-matching artifacts).

- **When to run:**
   - Run holdout evaluation after each reranker training run (and optionally after embed retrain).
   - Report synthetic/combined metrics and human-holdout metrics side-by-side for every experiment.

- **How to use results:**
   - Accept model changes only when gains also appear on the human holdout.
   - If synthetic metrics improve but holdout metrics do not, treat this as probable overfitting/leakage and revise data generation/splitting.

- **Operational rule:**
   - Keep `Training_Data/retriever_eval_queries_human_holdout.json` isolated from all training/candidate-building stages.
   - Use fixed `positive_ids` labels for holdout scoring (avoid filter-only labels).


## Evaluation Hardening Response (Post-Exp39)

The Exp39 gains are strong but may be inflated by synthetic-query easiness or leakage.  
**Code response plan (next implementation phase):**

1. TODO **Create held-out human eval set (50–100 queries)**
   - New file: `Training_Data/retriever_eval_queries_human_holdout.json`
   - Rules:
     - human-style wording only
     - no copied API tokens / constants from docs
     - fixed `positive_ids` only (no filter-only labels)
   - Keep this set excluded from all training stages.

2. TODO **Add leakage audit script**
   - New script: `Training_Data/audit_query_leakage.py`
   - Compute overlap signals between query text and positive chunk text:
     - exact token overlap ratio
     - API-token overlap ratio
     - heading substring hits
   - Output report: `% high-overlap queries`, top leakage examples, per-source breakdown (`human` vs `synthetic`).

3. TODO **Add split contamination guardrails**
   - New script: `Training_Data/validate_query_splits.py`
   - Enforce:
     - no duplicate/near-duplicate query text across train/test/holdout
     - no source-chunk-derived queries in holdout
     - optional lexical similarity threshold warnings
   - Fail CI/local run when violations exceed threshold.

4. TODO **Wire holdout eval into reranker pipeline manager**
   - Update `Train_Reranker_Model/0pipeline_manager.py` with a `--holdout-queries-file` option.
   - Run standard eval plus holdout-only eval in same pipeline invocation.
   - Save separate outputs for holdout:
     - `reranker_eval_summary_holdout.json`
     - `reranker_eval_summary_holdout.csv`

5. TODO **Run comparison protocol: Exp38 vs Exp39 on holdout**
   - Compare baseline and reranker on the same held-out human set.
   - Primary decision metrics: `MRR`, `Recall@5`, `NDCG@10`.
   - Accept Exp39 improvements only if gains persist on holdout.

### Acceptance Criteria for “real” gain

- Holdout uses only fixed `positive_ids` (no filter-derived positives).
- Holdout shows meaningful improvement over Exp38 (not just synthetic set).
- Leakage audit reports low overlap and no systemic query-copy patterns.
- Train/test/holdout contamination checks pass.



