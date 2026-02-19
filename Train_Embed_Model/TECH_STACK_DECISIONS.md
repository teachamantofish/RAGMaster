# Tech Stack Decisions — Implemented Choices (current)

This document records the *current* runtime and tuning choices implemented in this repository and why they were selected. It intentionally excludes legacy/fallback implementations and focuses on the stack we actively use for GPU-accelerated fine-tuning and the limited-step smoke runner.


## Tuniong decisions and rationale

### 1) Prefer CUDA (NVIDIA) for all heavy compute
- Why: CUDA gives native PyTorch acceleration and reliable performance on the RTX 5060 Ti.
- How we use it: model and tensors are moved to `torch.device('cuda')` when available. The smoke-runner and custom loops prefer CUDA and use autocast/GradScaler when fp16 is enabled.
- Impact: provides the largest single-step speedup for forward/backward, but requires careful memory management.

### 2) Mixed precision (fp16) via AMP + GradScaler
- Why: fp16 halves the memory required for activations and parameter copies (where supported) and often speeds up training on modern GPUs.
- How: we use PyTorch AMP (autocast) and `GradScaler` in the limited-run and training loops when `--fp16` is enabled.
- Impact on memory/perf: roughly 30–60% lower activation memory and faster tensor kernels on supported hardware. This allows larger batches or more activations per forward.
- Impact on embedding quality: with GradScaler and a conservative learning rate, fp16 typically preserves final quality close to fp32. However fp16 can be fragile (NaNs) if LR is too high or the model contains unstable ops. We added diagnostics and a one-time retry with fp16 disabled when NaNs are detected.

### 3) 8-bit optimizer (bitsandbytes) NOT DONE: requires Linux. 
- Why: optimizer state (Adam) requires two or more full-size tensors per parameter (exp_avg, exp_avg_sq). For large models this is the dominant memory cost. bitsandbytes stores those optimizer states in 8-bit, reducing memory by ~4x for optimizer state.
- How: when LoRA/adapters are used we prefer `bitsandbytes.optim.AdamW8bit` for adapter params if available; if not available we fall back to regular AdamW but with small parameter sets.
- Impact on memory/perf: large reduction in optimizer memory, enabling training that otherwise would OOM. It also tends to reduce host-to-device bandwidth for optimizer state. CPU cost is similar; bitsandbytes implements efficient kernels.
- Impact on quality: 8-bit optimizer state is an approximation but empirically often matches fp32 optimizer quality for reasonable LRs. Combine with fp16 weights for maximum memory reduction.

### 4) LoRA / PEFT adapters
- Why: full-weight fine-tuning of very large embedding models on a single 16GB GPU often fails due to optimizer memory. LoRA trains a small number of additional parameters (low-rank matrices) and keeps the base model frozen.
- How: we apply LoRA (via `peft`) to the underlying HF transformer submodule. The code injects adapters and ensures only adapter parameters are optimized.
- Impact on memory/perf: adapter-only training reduces gradient + optimizer memory proportionally to the adapter parameter count. This allows quick training iterations with tiny memory overhead and low optimizer state. Training time per step is lower than full-weight training because fewer parameters are updated.
- Impact on embedding quality: LoRA often reaches performance close to full-weight for many tasks, especially when domain shift is moderate, but may not always match full-weight quality for every dataset. It's a pragmatic tradeoff: much lower resource use for often-acceptable quality.

### 5) Prefetching: pre-tokenize + pinned-memory batches (new)

- What: We added a producer prefetcher that tokenizes upcoming micro-batches on CPU worker thread(s), converts tokenized dicts into pinned CPU tensors, and places them into a small FIFO queue. The training loop consumes these pre-pinned batches and moves them to GPU with non-blocking transfers.
- Why: This removes the tokenization/IO/GC work from the critical path between GPU steps. The GPU no longer waits for Python tokenization each iteration which smooths utilization and reduces idle spikes in Task Manager.
- Implemented functions / symbols:
  - `_to_pinned_tensor_dict(token_dict)` in `simple_training_test.py`: converts token dict entries into CPU pinned tensors for faster .to(device, non_blocking=True) transfers.
  - `_producer_thread_fn(batches)` in `simple_training_test.py`: background producer that tokenizes and enqueues pinned batches.
  - `prefetch_queue` (queue.Queue): small in-memory buffer of ready-to-run batches (default 4 batches).
  - `stop_producer` (threading.Event): used to gracefully stop the producer thread when the run completes.
  - Training loop changes inside `run_limited_training_steps`: it now fetches pre-tokenized batches from `prefetch_queue` and moves tensors to device using non_blocking transfers. It falls back to on-demand tokenization if the queue is empty.
- Expected benefits:
  - Much smoother GPU utilization with fewer idle spikes.
  - Lower per-step latency due to reduced CPU-side blocking between steps.
  - Small increase in host RAM usage (pinned memory) proportional to `PREFETCH_BATCHES * batch_size`.
- Trade-offs:
  - Slight code complexity and need to manage thread lifetime and pinned memory.
  - Pinned memory is a limited system resource — tune `PREFETCH_BATCHES` conservatively (default 4).

This change is conservative and opt-in within the limited-run harness; it does not affect other training paths unless they call `run_limited_training_steps`.

## 6) Additional tuning options implemented (1-5 requested)

The following options were implemented to further improve throughput and reduce CPU/GPU stalls. These are configurable via `TRAINING_CONFIG` in `config_embed_training.py`.

1) Use HF `AutoTokenizer` for batched tokenization
  - What: producer uses `transformers.AutoTokenizer` to tokenize lists of texts in bulk, which is faster and better optimized than repeated wrapper tokenization.
  - Where: `_producer_thread_fn` in `simple_training_test.py` will use `AutoTokenizer.from_pretrained(...)` when possible.

2) Prefetch workers
  - What: `TRAINING_CONFIG['PREFETCH_WORKERS']` controls how many producer workers to use. The default is 1 (thread). Using multiple processes avoids GIL constraints during tokenization.
  - Where: `simple_training_test.py` reads `PREFETCH_WORKERS` and starts the producer. For multiprocessing, future improvements can use `concurrent.futures.ProcessPoolExecutor`.

3) Torch.compile() optional wrapper
  - What: `TRAINING_CONFIG['USE_TORCH_COMPILE']` when True attempts to call `torch.compile(model)` to let PyTorch fuse kernels and reduce Python overhead.
  - Where: `embedding_finetuner.py::setup_model()` wraps the model when this flag is True.
  - Caveats: Requires PyTorch 2.x and can change memory/behavior; use only after verification.

4) Optional load_in_8bit + device_map
  - What: `TRAINING_CONFIG['USE_LOAD_IN_8BIT']` when True attempts a best-effort reload of the underlying HF model in 8-bit (`load_in_8bit=True, device_map='auto'`) and attaches it back to the SentenceTransformer wrapper.
  - Where: `embedding_finetuner.py::setup_model()` contains the loader logic.
  - Caveats: Requires `accelerate` and `bitsandbytes`; this is best-effort and falls back to normal load if unsupported.

5) Prefetch tuning and logging
  - New config keys: `PREFETCH_WORKERS` (int), `PREFETCH_BATCHES` (constant in harness, default 4). Adjust based on host RAM and CPU cores.

Files/functions to inspect for details:
- `simple_training_test.py`:
  - `run_limited_training_steps(...)` — now performs prefetching and consumes pinned batches.
  - `_to_pinned_tensor_dict(token_dict)` — converts token dict to pinned CPU tensors.
  - `_producer_thread_fn(batches)` — tokenizes using HF tokenizer when available and enqueues pinned batches.
  - `prefetch_queue` / `stop_producer` — queue and event for producer lifecycle.
- `embedding_finetuner.py`:
  - `setup_model()` — supports `USE_LOAD_IN_8BIT` and `USE_TORCH_COMPILE` flags.
- `config_embed_training.py`:
  - `TRAINING_CONFIG` now includes `USE_TORCH_COMPILE`, `USE_LOAD_IN_8BIT`, and `PREFETCH_WORKERS`.

Guidance for operators:
- Start conservative: keep `PREFETCH_BATCHES=4`, `PREFETCH_WORKERS=1` and monitor host memory. Increase `PREFETCH_WORKERS` to utilize more CPU cores for tokenization.
- If trying `USE_TORCH_COMPILE`, validate on a small run first and compare correctness and memory.
- If `USE_LOAD_IN_8BIT` is enabled, install `accelerate` and ensure your environment supports bitsandbytes; this path is optional and best-effort.

If you want, I can now:
- A) Replace the single-thread producer with a multiprocessing producer to bypass GIL and further speed tokenization.
- B) Reduce logging frequency to only sample DIAG every N steps to reduce CPU overhead.
- C) Implement a process-pool based producer and re-run a 20-step comparison.

Tell me which of A/B/C you prefer and I'll implement it and run a longer smoke test.

## Combined effect (LoRA + fp16 + bitsandbytes)
- Memory: 
  - fp16 reduces activation and parameter memory roughly 2x.
  - LoRA reduces gradients/optimizer state proportionally to the adapter size (often tiny: <0.1% of full model).
  - bitsandbytes reduces optimizer state (~4x less memory for Adam states).
  - Together these allow training on a 16GB GPU models that would otherwise need much more memory.
- Performance:
  - fp16 improves throughput on supported CUDA hardware.
  - Training adapters is faster per step due to fewer params to update.
  - bitsandbytes provides efficient optimizer updates with small memory footprint.
- Quality:
  - fp16 (with GradScaler) + bitsandbytes generally preserves final quality for adapter training.
  - LoRA may not perfectly match full-weight fine-tuning, but in practice yields strong results for embedding adaptation with much lower cost.

## Safety nets and diagnostics
- The limited-run includes:
  - Device-placement fixes (tokenized tensors moved to model device).
  - NaN/Inf diagnostics for embeddings (mean/std/min/max and any_nan checks).
  - One-time retry logic: if fp16 forward produces NaNs, the runner retries with fp16 disabled once.
  - OOM handling: if optimizer creation or step OOMs on CUDA, the runner can fallback to a CPU run to capture diagnostics rather than crash.

These measures reduce flakiness and provide clear logs to decide the next step.

## Practical recommendations / next steps
- Prefer LoRA + fp16 + bitsandbytes for single-GPU fine-tuning of large embedding models.
  - Start with LoRA rank r=8 and alpha=16 and monitor validation metrics.
  - If memory is still tight, reduce r to 4 or increase gradient accumulation.
- If you need absolutely maximum accuracy and have cluster/multi-GPU resources, use ZeRO/offload (DeepSpeed/Accelerate) to run full-weight fp32/fp16 training.
- Monitor for NaNs on early steps. If you see NaNs:
  - Lower learning rate and retry.
  - Disable fp16 and test forward/backward in fp32 to isolate the problem.
  - Verify the checkpoint integrity (run `diagnose_model_forward.py`).

## Files of interest
- `simple_training_test.py` — limited-run smoke function and LoRA integration.
- `embedding_finetuner.py` — CLI wiring (flags: `--fp16`, `--use_lora`, `--lora_r`, `--lora_alpha`).
- `diagnose_model_forward.py` — quick diagnostic helper used to validate model forward outputs.

## Tokenization pipeline (new)

We added a one-time tokenization pipeline to remove repeated host-side tokenization stalls and make repeated smoke/benchmark runs fast and deterministic.

- Script: `scripts/tokenize_triplets.py`
  - Purpose: tokenize the training triplets CSV once and save a tokenized dataset using `datasets.save_to_disk()`.
  - Config constants at the top of the script (edit as needed):
    - `MODEL_NAME` — which HF/SentenceTransformers tokenizer to use
    - `TRAIN_CSV` — input CSV filename (expects `anchor`, `positive`, `negative` columns)
    - `MAX_LENGTH` — token truncation/padding length
    - `PADDING` — padding strategy (default `max_length` in the script)
    - `NUM_PROC` — parallel workers for `datasets.map()` (set smaller on Windows)
    - `BATCH_SIZE` — tokenization batch size
  - Outputs:
    - tokenized dataset directory: `tokenized_train/` (saved via `save_to_disk`)
    - `metadata.json` in the tokenized dir with tokenizer name, max_length, padding, num_proc, batch_size, `avg_anchor_length`, `p90_anchor_length`, `p95_anchor_length`, and `num_examples`.

- Why: tokenizing once eliminates the CPU tokenization cost from the training critical path. The saved dataset can be loaded quickly and consumed by the training loop with minimal overhead.

- Best practices:
  - Use `num_proc` > 1 on Linux for parallel tokenization; on Windows prefer `num_proc=1` or run the script as a script (not interactive) to avoid multiprocessing issues.
  - Choose `MAX_LENGTH` using the saved `metadata.json` (avg/p90/p95) — a common approach is to set `max_length` to the 95th percentile rounded to a multiple of 8.
  - Prefer saving tokenized arrays (no torch tensors) and performing dynamic padding in the DataLoader via a DataCollator if you want smaller disk size.

- Finetuner check: `embedding_finetuner.py` now validates that `TOKENIZED_DATA_DIR` exists before running `fine_tune_model()` or `--action smoke_steps`. If missing, it aborts with a clear message and a suggested command:
  ```powershell
  python scripts/tokenize_triplets.py
  ```

- How to run the tokenizer (example):
  ```powershell
  .venv\Scripts\Activate.ps1
  python scripts/tokenize_triplets.py
  ```

Adding this pipeline significantly reduces per-step variance and host-side stalls. After tokenizing, re-run your 5-step smoke test to confirm per-step time and variance improvements.

## Prepadding rationale

Short answer: pad to 88 is a good choice — it’s simple, aligned to hardware (88 is divisible by 8), and will capture ~95%+ of your examples while cutting wasted work vs 256 by a large margin.

Why pre-padding to 88 is sensible here

Your stats (avg ≈56, p90=80, p95=84) show almost all examples are ≤88, so truncation loss will be tiny.
Pre-padding to a fixed, small length gives predictable peak memory and removes per-batch padding work at training time. That helps the prefetch/pinned-tensor pipeline you already implemented: you can create CPU-pinned tensors once (already padded), queue them, and move them to GPU non-blocking. That minimizes CPU overhead and GPU stalls.
88 is a multiple of 8 (good for some GPU kernels) so performance is fine; if you wanted to be ultra-conservative 96 is also fine — but 88 is compact and safe given your p95.
Tradeoffs vs dynamic padding (DataCollator / batch-longest)

Compute: Dynamic padding (pad-to-batch-longest) will reduce the total tokens processed slightly more than fixed-88 when batches have many very-short examples, because it never does padding to the fixed maximum. So it can reduce FLOPS and slightly reduce runtime.
Memory (peak): Dynamic padding can reduce per-batch memory if batches are skewed short. But peak memory is driven by the largest batch member — if you still include some long examples in a batch, peak won't drop that much. Fixed-88 gives deterministic peak which helps avoid OOM surprises.
Overhead/complexity: Dynamic padding requires a collator on every batch (CPU work) and potentially varies tensor shapes each batch. That adds small per-batch CPU cost and slightly more complexity to keep tensors pinned and prefetch working efficiently. If you already moved tokenization off the hot path and have a pinned-buffer prefetch pipeline, pre-padding is simpler and likely faster in practice.
Implementation safety: Pre-padding lets you create the dataset once, reuse it, and keeps the training loop simpler and more deterministic.
Recommendation

Given your observed lengths, pre-padding to 88 is the pragmatic, low-complexity win: it reduces compute and memory dramatically vs 256, keeps the prefetch pipeline simple, and only truncates a tiny tail of examples.
If later you want to squeeze a bit more compute out, you can switch to a DataCollatorWithPadding to use dynamic per-batch padding; but test to confirm the extra CPU collate overhead doesn’t offset the token savings on your hardware.





=====================

combined_data = dataset1 + dataset2 # Combine datasets and interleave


random.shuffle(combined_data)    	# Reshuffle after each epoch





### Exp 35 — Fixed broken codenames in chunk content (DONE)

**What:** Fixed 493 line-break artifacts in chunk content where FDK constants were split by spaces during web crawling (e.g. `FE_BadParamete r` → `FE_BadParameter`). User also reviewed and edited ~6500 friendly name expansions in the CSV. Content quality improved but no pipeline code changes.

**Results (alpha sweep, best alpha = 0.65):**

| Metric | Exp 34 | Exp 35 | Delta |
|--------|--------|--------|-------|
| Best alpha | 0.65 | 0.65 | — |
| Blended MRR (all) | 0.810 | **0.811** | **+0.1pp** |
| ID-test MRR | 0.836 | 0.835 | -0.1pp |
| ID-test R@5 | 0.891 | **0.896** | **+0.5pp** |
| Top1% | 0.728 | **0.733** | **+0.5pp** |
| Val accuracy | 0.933 | 0.934 | +0.1pp |

**Key observations:**
- Essentially flat — within noise margins. Fixing broken tokens helped content quality but the cross-encoder was already handling broken subwords via WordPiece tokenization
- The real payoff from fixed codenames + friendly name enrichment will come when bi-encoder embeddings are re-generated (retriever improvement, not reranker)
- Confirms the reranker is near plateau with current retriever quality — further gains require improving the top-200 retrieval slate

---

## Final Complete Experiment Summary (All Phases)

| Phase | Experiments | Model | Best MRR (test) | vs Baseline |
|---|---|---|---|---|
| 1 (Hyperparams) | 0-9 | Qwen3-Reranker-0.6B (FSC) | 0.495 | -14.2% |
| 2.1 (Listwise) | 11-13 | Qwen3-Reranker-0.6B (FSC) | 0.190 | -67.1% |
| 2 (Freeze) | 14 | Qwen3-Reranker-0.6B (FSC) | 0.070 | -87.9% |
| Discovery | 15 | Qwen3-Reranker-0.6B (CLM) | 0.092 | -84.1% |
| 3 (Model swap) | 16-28 | ms-marco-MiniLM-L-6-v2 | 0.685* | +18.7% |
| 4 (Interpolation) | 29 | blend α=0.73 | 0.707* | +22.5% |
| **5 (More data)** | **30** | **51 train queries + α=0.67** | **0.770** | **+30.5%** |
| **6 (Leakage fix)** | **31-32** | **16 ID-only train + α=0.70** | **0.748 (ID: 0.762)** | **+26.0% (ID: +19.8%)** |
| **6 (Relabel)** | **33** | **43 ID-labeled train + α=0.80** | **0.792 (ID: 0.824)** | **+34.1% (ID: +29.5%)** |
| **6 (Expand)** | **34** | **69 ID-labeled train + α=0.65** | **0.810 (ID: 0.836)** | **+37.2% (ID: +31.4%)** |
| **7 (Content fix)** | **35** | **Fixed broken codenames + α=0.65** | **0.811 (ID: 0.835)** | **+37.3% (ID: +31.3%)** |

\* Phases 3-4 evaluated on 188 test queries; Phase 5 on 148 (harder subset).

### Final Lessons Learned

1. **Model architecture > hyperparameters**: 15 experiments of hyperparameter tuning on the wrong model architecture yielded nothing. One model swap gave a +18.7% improvement.
2. **Verify model compatibility**: The Qwen3-Reranker was a causal LM loaded as ForSequenceClassification — its entire pre-trained capability was being discarded.
3. **Start with established baselines**: MS MARCO cross-encoders are the standard for re-ranking. Should have been the starting point.
4. **Zero-shot is a strong baseline**: The ms-marco model beat our baseline with zero-shot (MRR 0.650). Pre-training data quality matters more than domain-specific fine-tuning.
5. **Small models can win**: 22M parameter MiniLM dramatically outperformed 600M parameter Qwen3 — because the small model was trained correctly for the task.
6. **Don't discard retriever signal**: Blending reranker + baseline outperforms either alone. The reranker complements the retriever, it doesn't replace it.
7. **Score interpolation is free precision**: No retraining, no extra inference — just weighted arithmetic on already-computed scores.
8. **Training data quantity matters enormously**: Going from 11 → 51 queries (2,200 → 13,894 pairs) was the single largest improvement. The reranker's quality is directly proportional to how many diverse query patterns it has seen during training.
9. **Watch for label leakage**: Filter-based positive labels (keyword contains) inflated MRR by +30pp on filter queries. The model learned to score keyword-containing candidates higher — a shortcut, not real relevance. Removing those queries improved the honest metric by +3.5pp. Always split metrics by label type.
10. **Report honest metrics**: A single blended number can hide that your model is great at the easy thing and mediocre at the hard thing. Split reporting (Exp 31) should be permanent.
11. **Manual labeling pays off**: Converting 27 noisy filter-labeled queries to hand-picked chunk IDs produced +6.2pp on ID-test MRR — the single largest gain in the project. Clean labels > more labels.
12. **Content fixes help retrieval, not reranking**: Fixing 493 broken codenames in chunk text had negligible reranker impact because the cross-encoder's subword tokenizer was already handling broken tokens. The fix will matter more for bi-encoder embeddings where exact token matching drives similarity.


