# Overview:
# Use BERTScore and SummaC as complementary, asymmetric evaluators.
# - BERTScore measures semantic similarity and coverage: how well the summary captures the meaning of the source chunk.
#   It is a scalar quality signal used for ranking, regression tracking, and identifying weak or underspecified summaries.
# - SummaC measures factual consistency: whether each summary sentence is entailed by the source chunk.
#   It surfaces hallucinations and unsupported claims so you can inspect low-confidence summaries.
# Workflow:
#   1) Run BERTScore on all summaries to assess semantic quality.
#   2) Run SummaC as a second-pass faithfulness check.
# Interpretation:
#   - High BERTScore + pass SummaC → good, trustworthy summary.
#   - High BERTScore + fail SummaC → fluent but unfaithful (hallucination risk).
#   - Low BERTScore + pass SummaC → faithful but incomplete or overly compressed.
# Do not average the scores; use BERTScore for quality ranking and SummaC as a correctness guardrail.

# Summary evaluation settings (semantic similarity / coverage)
BERT_MODEL_TYPE = "roberta-large"          # Large En-only encoder; best default for semantic similarity in summaries (use neobert someday)
BERT_LANG = "en"                           # English-only evaluation; avoids multilingual embedding dilution
BERT_RESCALE_WITH_BASELINE = False         # Normalize scores for cross-run and cross-model comparability
BERT_HIGH_THRESHOLD = 0.80                 # At/above this, treat BERTScore as "high" coverage
BERT_LOW_THRESHOLD = 0.65                  # At/below this, treat BERTScore as "low" coverage

# SummaC evaluation settings (factual consistency / faithfulness)
# Often not useful for API/SDK docs, especially tables and examples. 

SUMMAC_AVAILABLE = False                  # Flip to False if SummaC dependencies are unavailable
SUMMAC_MODEL_TYPE = "vitc"                # Default SummaC zero-shot model (sentence-level entailment)
SUMMAC_GRANULARITY = "sentence"           # Request sentence-level evidence scores from SummaC
SUMMAC_EVIDENCE_SCOPE = "chunk_only"      # Restrict evidence to the source chunk used to generate the summary
SUMMAC_SKIP_TABLE_CHUNKS = True           # Skip evaluation of table chunks: Not implemented.
SUMMAC_SKIP_EXA_CHUNKS = True             # Skip evaluation of code example chunks: Not implemented.

# A plain mean treats every sentence equally and hides two failure modes we care about:
# (1) a single hallucinated line can be drowned out by many faithful sentences, and 
# (2) noisy outliers (very high or very low) can whiplash the average. The percentile + trimmed‑mean pair tackles those directly:

# The percentile checks how bad the weaker tail actually is (e.g., “even the 20th percentile dips below 0.5”), 
# which is a stronger signal for “one paragraph is hallucinating” than seeing a mean of 0.78.
# The trimmed mean keeps most of the distribution but chops off the extremes so one glitchy sentence
# doesn’t invalidate an otherwise solid summary, while the average still reflects overall faithfulness.
# If you’re not acting on those failure cases—say you only log a single scalar per chunk—then the extra 
# metrics may not add much. But if you want to flag “pass overall but risky in one section,” the percentile 
# gives you a handle to do that, and the trimmed mean serves as a more robust primary score than the straight mean. 
# A compromise is to keep the mean as SUMMAC_PRIMARY_AGGREGATE and still log the percentile so humans can inspect low tails when needed.

SUMMAC_PERCENTILE = 0.20                  # Percentile used when aggregating sentence scores (e.g., 20th)
SUMMAC_TRIM_RATIO = 0.10                  # Fraction trimmed from each tail when computing trimmed mean (discards extremes lows)
SUMMAC_PRIMARY_AGGREGATE = "trimmed_mean"  # Primary scalar to log/compare (mean/percentile/trimmed_mean)




