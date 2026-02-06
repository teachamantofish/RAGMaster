import os
import copy
import json
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config.embedconfig import *
from common.utils import (get_csv_to_process, setup_global_logger)

_DATA_CONTEXT = get_csv_to_process()
DATASET_ROOT = _DATA_CONTEXT['cwd']
embedfile = DATASET_ROOT / "a_chunks.json"  # Get json file with chunks to embed.
post_embed_file = embedfile.with_name("a_chunks_postembedding.json")
ADAPTER_DIR = (DATASET_ROOT / Path(ADAPTER_PATH)).resolve() if ADAPTER_PATH else None

# Set up global logger with script-specific CSV header; overwrite existing log
script_base = os.path.splitext(os.path.basename(__file__))[0]
LOG_HEADER = ["Date", "Level", "Message", "Chunk ID"]
logger = setup_global_logger(script_name=script_base, log_level=EMBED_LOG_LEVEL, headers=LOG_HEADER)


def _attach_lora_adapter(sentence_model: SentenceTransformer, adapter_dir: Path) -> None:
    """Load PEFT LoRA weights into the underlying transformer inside SentenceTransformer."""
    try:
        from peft import PeftModel
    except Exception as exc:  # pragma: no cover - import guard
        logger.error(
            "Adapter path %s provided but 'peft' is unavailable (%s). Install peft to use fine-tuned adapters.",
            adapter_dir,
            exc,
        )
        sys.exit(1)

    first_module = None
    base_transformer = None
    if hasattr(sentence_model, "_first_module"):
        try:
            first_module = sentence_model._first_module()
            base_transformer = getattr(first_module, "auto_model", None)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to access base transformer module for adapter loading: %s", exc)
            base_transformer = None

    if base_transformer is None:
        logger.error("Unable to locate underlying transformer to attach adapter; skipping adapter load.")
        return

    logger.info("Loading LoRA adapter weights from %s", adapter_dir)
    try:
        peft_model = PeftModel.from_pretrained(base_transformer, adapter_dir.as_posix(), is_trainable=False)
        # Replace the auto_model reference so downstream encode() uses the adapted weights.
        if first_module is None:
            logger.error("SentenceTransformer structure changed unexpectedly; adapter not applied.")
            return
        first_module.auto_model = peft_model
        logger.info("LoRA adapter applied successfully; embeddings will use fine-tuned weights.")
    except Exception as exc:
        logger.error("Failed to load adapter from %s: %s", adapter_dir, exc)
        raise

def _fmt_size(n_bytes: int) -> str:
    """Human-readable file size formatter."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n_bytes)
    for u in units:
        if size < 1024.0 or u == units[-1]:
            return f"{size:.1f} {u}"
        size /= 1024.0

def update_provenance_with_embedding():
    """Load a_provenance.json, add embedding details, and save back."""
    prov_path = DATASET_ROOT / "a_provenance.json"
    
    # Load existing provenance
    try:
        with open(prov_path, "r", encoding="utf-8") as f:
            provenance = json.load(f)
    except FileNotFoundError:
        if 'PROVENANCE_REQUIRED' in globals() and PROVENANCE_REQUIRED:
            logger.error(f"Provenance file not found at {prov_path}. Aborting.")
            sys.exit(1)
        else:
            logger.warning(f"Provenance file not found at {prov_path}. Creating a new one.")
            provenance = {}
    
    # Update embed details (fill in existing keys)
    provenance["embed"] = {
        "basemodel": EMBED_MODEL,
        "adaptermodel": ADAPTER_PATH,
        # Precision and dimension settings captured for reproducibility and auditing
        "compute_precision": EMBED_COMPUTE_PRECISION,
        "output_precision": EMBED_OUTPUT_PRECISION,
        "vector_dim": EMBED_VECTOR_DIM,
    }
    
    # Save back
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Updated provenance file at {prov_path}")

with open(embedfile, 'r', encoding='utf-8') as f:
    chunks = json.load(f)
logger.info("Loaded chunks from %s; count=%d", embedfile, len(chunks) if hasattr(chunks, '__len__') else 0)

# Optional: select a subset of chunks for a quick test run
def _get_max_chunks_from_env(default_val):
    val = os.getenv("EMBED_MAX_CHUNKS")
    if not val:
        return default_val
    try:
        n = int(val)
        return n if n > 0 else None
    except Exception:
        logger.warning(f"Ignoring invalid EMBED_MAX_CHUNKS env value: {val}")
        return default_val

def _get_sample_mode_from_env(default_val):
    val = os.getenv("EMBED_SAMPLE_MODE")
    if not val:
        return default_val
    v = val.strip().lower()
    if v in ("head", "random"):
        return v
    logger.warning(f"Ignoring invalid EMBED_SAMPLE_MODE env value: {val}")
    return default_val

MAX_N = _get_max_chunks_from_env(globals().get('MAX_EMBED_CHUNKS', None))
SAMPLE_MODE = _get_sample_mode_from_env(globals().get('CHUNK_SAMPLE_MODE', 'head'))
SAMPLE_SEED = globals().get('CHUNK_SAMPLE_SEED', 42)

selected_indices = None
if isinstance(chunks, list) and MAX_N is not None:
    total = len(chunks)
    n = min(MAX_N, total)
    if n < total:
        if SAMPLE_MODE == 'random':
            import random
            rng = random.Random(SAMPLE_SEED)
            selected_indices = set(sorted(rng.sample(range(total), n)))
            logger.info(f"Sampling {n} chunks uniformly at random (seed={SAMPLE_SEED}) out of {total}")
        else:
            selected_indices = set(range(n))
            logger.info(f"Taking first {n} chunks (head mode) out of {total}")
    else:
        logger.info("Sampling configured but N >= total; embedding all chunks")
else:
    logger.info("Embedding all available chunks (no sampling limit configured)")

"""Enforce CUDA usage only.
If CUDA is unavailable, exit with a non-zero code and a clear error message.
"""
if not torch.cuda.is_available():
    msg = (
        "CUDA is required for embeddings but was not found. "
        "Please install a CUDA-enabled PyTorch build and ensure an NVIDIA GPU is available."
    )
    logger.error(msg)
    sys.exit(1)

# Prefer a specific CUDA device if configured; otherwise use the first visible GPU
try:
    device_index = None
    if 'DEVICE_ID' in globals() and DEVICE_ID is not None:
        # Validate index
        count = torch.cuda.device_count()
        if DEVICE_ID < 0 or DEVICE_ID >= count:
            logger.error(f"Configured DEVICE_ID={DEVICE_ID} is out of range. Visible GPUs: {count}")
            sys.exit(1)
        device_index = DEVICE_ID
    else:
        device_index = 0

    device = f"cuda:{device_index}" if device_index is not None else "cuda"
    gpu_name = torch.cuda.get_device_name(device_index)
    logger.info(f"Using GPU {device_index}: {gpu_name}")
except Exception as e:
    logger.error(f"Failed to select/query CUDA device: {e}")
    sys.exit(1)

# Precision controls: map config strings to torch dtypes and configure math paths
def _dtype_from_string(name: str):
    name = (name or "").strip().lower()
    if name in ("fp32", "float32"): return torch.float32
    if name in ("fp16", "float16", "half"): return torch.float16
    if name in ("bf16", "bfloat16"): return torch.bfloat16
    if name == "tf32": return torch.float32  # tensors remain float32; TF32 enabled via flags below
    raise ValueError(f"Unsupported precision string: {name}")

try:
    compute_dtype = _dtype_from_string(EMBED_COMPUTE_PRECISION)
except Exception as e:
    logger.error(f"Invalid EMBED_COMPUTE_PRECISION '{EMBED_COMPUTE_PRECISION}': {e}")
    sys.exit(1)

try:
    output_dtype = _dtype_from_string(EMBED_OUTPUT_PRECISION)
except Exception as e:
    logger.error(f"Invalid EMBED_OUTPUT_PRECISION '{EMBED_OUTPUT_PRECISION}': {e}")
    sys.exit(1)

# Configure TF32 acceleration for float32 compute if requested and supported
try:
    if ENABLE_TF32 and EMBED_COMPUTE_PRECISION.strip().lower() in ("tf32", "float32", "fp32"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # For PyTorch >= 2.0, helps enable faster matmul in float32 paths
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")  # or "medium" if you want less aggressive
        logger.info("TF32 acceleration enabled for float32 compute paths")
except Exception as _e:
    # Non-fatal: log at debug level if needed
    pass

# Load the model directly from HuggingFace name instead of local path
embed_model = SentenceTransformer(EMBED_MODEL, device=device)

if ADAPTER_DIR:
    if ADAPTER_DIR.exists():
        _attach_lora_adapter(embed_model, ADAPTER_DIR)
    else:
        logger.warning(
            "Configured adapter path '%s' not found at %s; falling back to base model weights.",
            ADAPTER_PATH,
            ADAPTER_DIR,
        )

# Move model parameters to the desired compute dtype for faster inference when applicable
try:
    embed_model = embed_model.to(dtype=compute_dtype, device=device)
    logger.info(f"Embedding compute precision set to {EMBED_COMPUTE_PRECISION}")
except Exception as e:
    logger.warning(f"Could not set model dtype to {EMBED_COMPUTE_PRECISION}: {e}")

# Defensive check: ensure the model is targeting CUDA (no silent CPU fallback)
target_device = getattr(embed_model, "device", getattr(embed_model, "_target_device", None))
td_str = str(target_device) if target_device is not None else "unknown"
if not td_str.startswith("cuda"):
    logger.error(f"Model target device is not CUDA (got: {td_str}). Aborting.")
    sys.exit(1)

try:
    emb_dim = embed_model.get_sentence_embedding_dimension()
    logger.info(f"This model's embedding dimension: {emb_dim}")
except Exception:
    pass

# Compare with configured dimension, warn or enforce as configured
try:
    cfg_dim = EMBED_VECTOR_DIM
    if isinstance(cfg_dim, int) and cfg_dim > 0 and emb_dim is not None:
        if cfg_dim != emb_dim:
            msg = (
                f"Configured EMBED_VECTOR_DIM={cfg_dim} does not match model-reported dimension {emb_dim}. "
                "No automatic dimensionality change is applied in this script. Ensure downstream consumers "
                "(e.g., pgvector column and indexes) match the actual dimension, or add an explicit projection/"
                "reduction step to produce the configured dimension before persistence."
            )
            if 'ENFORCE_EMBED_VECTOR_DIM' in globals() and ENFORCE_EMBED_VECTOR_DIM:
                logger.error(msg)
                sys.exit(1)
            else:
                logger.warning(msg)
except Exception as _e:
    # Non-fatal: if config missing or unexpected type
    pass

logger.info(f"Loaded {EMBED_MODEL} embedding model on {device}")

# Apply embeddings to each chunk, respecting precision settings
embedded_count = 0
skipped_unselected = 0
processed_candidates = 0
for idx, chunk in enumerate(chunks):
    if ENABLE_EMBEDDING:
        # If a sampling subset is active and this index isn't selected, skip it
        if selected_indices is not None and idx not in selected_indices:
            skipped_unselected += 1
            continue
        processed_candidates += 1
        if str(chunk.get('embedding')).lower() == 'false':
            logger.info(f"Embedding skipped for {chunk['id']} (tokens: {chunk['token_count']}) because embedding is 'false'.")
            # Normalize representation to None when JSON contains 'false' for downstream consumers
            chunk['embedding'] = None
            continue
        logger.info(f"Embedding {chunk['id']}. Tokens: {chunk['token_count']}")
        try:
            # Use tensor output to preserve/control dtype, then cast to requested output dtype
            # Note: compute dtype is controlled by model param dtype; autocast is generally not needed here
            emb_tensor = embed_model.encode(
                chunk['content'],
                convert_to_tensor=True,
                device=device,
                normalize_embeddings=NORMALIZE_EMBEDDINGS,
            )
            # Ensure on GPU then cast to desired output dtype
            if emb_tensor.device.type != 'cuda':
                emb_tensor = emb_tensor.to('cuda')
            # Extra safety: normalize if library version didn't apply it
            if NORMALIZE_EMBEDDINGS:
                try:
                    emb_tensor = F.normalize(emb_tensor, p=2, dim=-1)
                except Exception:
                    pass
            emb_tensor = emb_tensor.to(dtype=output_dtype)
            # Move to CPU for JSON serialization
            chunk['embedding'] = emb_tensor.detach().cpu().tolist()
            if chunk['embedding'] is None:
                logger.error(f"Embedding API returned None for {chunk['id']} (tokens: {chunk['token_count']})")
            else:
                logger.info(f"Embedding generated for {chunk['id']} (tokens: {chunk['token_count']})")
                embedded_count += 1

            # Also embed summaries for better high-level retrieval
            # These are stored as separate fields for pgvector (summary-specific embeddings)
            for summary_key, embed_key in (
                ("chunk_summary", "embedding_summary_chunk"),
                ("page_summary", "embedding_summary_page"),
            ):
                summary_text = chunk.get(summary_key) or ""
                if not summary_text:
                    chunk[embed_key] = None  # no summary -> no summary embedding
                    logger.info(f"Summary embedding skipped for {chunk['id']} ({summary_key}: empty)")
                    continue
                summary_tensor = embed_model.encode(
                    summary_text,
                    convert_to_tensor=True,
                    device=device,
                    normalize_embeddings=NORMALIZE_EMBEDDINGS,
                )
                if summary_tensor.device.type != 'cuda':
                    summary_tensor = summary_tensor.to('cuda')
                if NORMALIZE_EMBEDDINGS:
                    try:
                        summary_tensor = F.normalize(summary_tensor, p=2, dim=-1)
                    except Exception:
                        pass
                summary_tensor = summary_tensor.to(dtype=output_dtype)
                chunk[embed_key] = summary_tensor.detach().cpu().tolist()
                logger.info(f"Summary embedding generated for {chunk['id']} ({summary_key})")

            # Positive proof log: confirm all three embeddings exist for this chunk
            if (
                chunk.get("embedding") is not None
                and chunk.get("embedding_summary_chunk") is not None
                and chunk.get("embedding_summary_page") is not None
            ):
                logger.info(
                    f"Embeddings complete for {chunk['id']}: content + chunk_summary + page_summary"
                )
        except Exception as e:
            # If any embedding step fails, keep fields explicit for downstream handling
            chunk['embedding'] = None
            chunk['embedding_summary_chunk'] = None
            chunk['embedding_summary_page'] = None
            logger.error(f"LLM embedding error for {chunk['id']}: {e}")
    else:
        chunk['embedding'] = None
        logger.info(f"Embedding skipped for {chunk['id']} (tokens: {chunk['token_count']}) because ENABLE_EMBEDDING is False.")

# Save a copy of the input JSON with embeddings removed (post-embedding JSON should be metadata-only)
stripped_chunks = copy.deepcopy(chunks)
stripped = 0
for ch in stripped_chunks:
    if 'embedding' in ch and ch['embedding'] is not None:
        ch['embedding'] = None
        stripped += 1
    if 'embedding_summary_chunk' in ch and ch['embedding_summary_chunk'] is not None:
        ch['embedding_summary_chunk'] = None
        stripped += 1
    if 'embedding_summary_page' in ch and ch['embedding_summary_page'] is not None:
        ch['embedding_summary_page'] = None
        stripped += 1
with open(post_embed_file, 'w', encoding='utf-8') as f:
    json.dump(stripped_chunks, f, indent=JSON_INDENT, ensure_ascii=False)
logger.info(f"Saved stripped post-embedding JSON (no vectors) to {post_embed_file} | stripped fields: {stripped}")
try:
    _json_stripped_size = post_embed_file.stat().st_size
    logger.info(f"Current JSON size (stripped): {_fmt_size(_json_stripped_size)}")
except Exception:
    _json_stripped_size = None

# Log a brief summary of what we processed
try:
    total_chunks = len(chunks) if isinstance(chunks, list) else 0
    logger.info(
        "Embedding summary -> total: %s | candidates: %s | embedded: %s | skipped-unselected: %s",
        total_chunks, processed_candidates, embedded_count, skipped_unselected,
    )
except Exception:
    pass

# === Optional: Save embeddings to Parquet sidecar for compact, fast I/O ===
def save_embeddings_to_parquet(chunks_list, parquet_path: Path, emb_dim: int, compression: str = "zstd", row_group_size: int | None = None):
    """
    Write embeddings to a Parquet sidecar file with schema: [id: string, embedding: fixed_size_list<float32, emb_dim>].

    Rationale and benefits:
    - JSON is verbose for large float arrays (each float rendered as text + commas/newlines), which inflates file size
      and slows I/O. Parquet stores binary floats with optional compression (snappy/zstd), dramatically reducing size
      and speeding read/write.
    - This is a storage/serialization change only; it is lossless for retrieval quality because we preserve float32
      values exactly (no quantization).
    - Keeping embeddings outside the JSON keeps the metadata file small and easy to diff/inspect, while Parquet handles
      the heavy numeric payload efficiently.

    Implementation details:
    - We use a fixed-size list column for 'embedding' to enforce consistent vector length (emb_dim) across rows.
    - We cast to float32 per pgvector’s storage and common downstream expectations.
    - If pyarrow is not installed, we emit a clear message with install guidance.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as e:
        raise RuntimeError(
            "pyarrow is required to write Parquet embeddings. Install it with: pip install pyarrow"
        ) from e

    ids = []
    vectors = []
    summary_vectors = []
    page_vectors = []
    for ch in chunks_list:
        emb = ch.get('embedding')
        emb_sum = ch.get('embedding_summary_chunk')
        emb_page = ch.get('embedding_summary_page')
        cid = ch.get('id')
        if cid is None:
            continue
        if emb is None:
            continue
        # Only accept list/tuple/ndarray; skip strings/bools or malformed entries
        if not isinstance(emb, (list, tuple, np.ndarray)):
            logger.warning(f"Skipping embedding for id={cid}: non-numeric type {type(emb).__name__}")
            continue
        try:
            vec = np.asarray(emb, dtype=np.float32)
        except Exception as conv_e:
            logger.warning(f"Skipping embedding for id={cid}: cannot convert to float32 ({conv_e})")
            continue
        if vec.ndim != 1 or (emb_dim is not None and vec.shape[0] != emb_dim):
            logger.warning(f"Skipping embedding for id={cid}: wrong shape {vec.shape}, expected ({emb_dim},)")
            continue
        ids.append(str(cid))
        vectors.append(vec)

        # Keep summaries aligned by id; if missing or invalid, store None
        def _coerce_or_none(val, label):
            if val is None:
                return None
            if not isinstance(val, (list, tuple, np.ndarray)):
                logger.warning(f"Skipping {label} for id={cid}: non-numeric type {type(val).__name__}")
                return None
            try:
                v = np.asarray(val, dtype=np.float32)
            except Exception as conv_e:
                logger.warning(f"Skipping {label} for id={cid}: cannot convert to float32 ({conv_e})")
                return None
            if v.ndim != 1 or (emb_dim is not None and v.shape[0] != emb_dim):
                logger.warning(f"Skipping {label} for id={cid}: wrong shape {v.shape}, expected ({emb_dim},)")
                return None
            return v

        summary_vectors.append(_coerce_or_none(emb_sum, "embedding_summary_chunk"))
        page_vectors.append(_coerce_or_none(emb_page, "embedding_summary_page"))

    if not vectors:
        logger.warning("No embeddings available to write to Parquet; skipping.")
        return

    # Flatten vectors to a single values array, then form a FixedSizeListArray
    flat = np.concatenate(vectors).astype(np.float32, copy=False)
    values = pa.array(flat, type=pa.float32())

    # Prefer FixedSizeList if available, else fall back to list_ with list_size
    try:
        emb_array = pa.FixedSizeListArray.from_arrays(values, emb_dim)
        emb_type = pa.list_(pa.float32(), list_size=emb_dim)
    except AttributeError:
        # Older Arrow versions
        emb_type = pa.list_(pa.float32(), list_size=emb_dim)
        # Build offsets for a regular ListArray as fallback
        offsets = pa.array([i * emb_dim for i in range(len(vectors) + 1)], type=pa.int32())
        emb_array = pa.ListArray.from_arrays(offsets, values)

    arrays = [pa.array(ids, type=pa.string()), emb_array]
    names = ["id", "embedding"]

    # Optional summary embeddings (preserve row alignment; nulls allowed)
    def _to_list_array(vecs):
        if not vecs:
            return None
        vals = []
        offsets = [0]
        for v in vecs:
            if v is None:
                offsets.append(offsets[-1])
                continue
            vals.append(v)
            offsets.append(offsets[-1] + emb_dim)
        if not vals:
            return None
        flat_local = np.concatenate(vals).astype(np.float32, copy=False)
        values_local = pa.array(flat_local, type=pa.float32())
        offsets_arr = pa.array(offsets, type=pa.int32())
        return pa.ListArray.from_arrays(offsets_arr, values_local)

    sum_array = _to_list_array(summary_vectors)
    if sum_array is not None:
        arrays.append(sum_array)
        names.append("embedding_summary_chunk")

    page_array = _to_list_array(page_vectors)
    if page_array is not None:
        arrays.append(page_array)
        names.append("embedding_summary_page")

    table = pa.Table.from_arrays(arrays, names=names)

    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet_path.as_posix(), compression=compression, row_group_size=row_group_size)
    logger.info(f"Wrote {len(ids)} embeddings to Parquet: {parquet_path} (dim={emb_dim}, codec={compression})")

# If requested, emit a Parquet sidecar and strip embeddings from JSON to keep it lean
try:
    if USE_PARQUET:
        # Determine output paths and dimension
        parquet_file = DATASET_ROOT / PARQUET_FILENAME
        # Infer dimension (we logged earlier too)
        try:
            emb_dim = embed_model.get_sentence_embedding_dimension()
        except Exception:
            # Fallback: find first non-empty embedding
            emb_dim = None
            for ch in chunks:
                if isinstance(ch.get('embedding'), list):
                    emb_dim = len(ch['embedding'])
                    break
        if not emb_dim:
            raise ValueError("Could not determine embedding dimension for Parquet output")

        save_embeddings_to_parquet(chunks, parquet_file, emb_dim, PARQUET_COMPRESSION, PARQUET_ROW_GROUP_SIZE)

        # Size summary for stripped JSON + Parquet
        try:
            _parquet_size = parquet_file.stat().st_size if parquet_file.exists() else 0
            if _json_stripped_size is not None:
                logger.info(
                    "Size summary -> Stripped JSON: %s | Parquet: %s | Stripped+Parquet total: %s",
                    _fmt_size(_json_stripped_size),
                    _fmt_size(_parquet_size),
                    _fmt_size((_json_stripped_size or 0) + _parquet_size),
                )
        except Exception:
            pass
except Exception as e:
    logger.error(f"Parquet export step failed: {e}")

# Finally, update provenance with the run's embedding settings
try:
    update_provenance_with_embedding()
except SystemExit:
    # Propagate intentional exit (e.g., missing provenance file as requested)
    raise
except Exception as e:
    logger.error(f"Failed to update provenance: {e}")
