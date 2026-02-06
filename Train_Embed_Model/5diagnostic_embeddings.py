# Diagnostic script to check fine-tuned embedding model
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from scripts.config_training_embed import *
from scripts.custom_logger import setup_global_logger

script_base = Path(__file__).stem
LOG_HEADER = ["Item", "Tuned model", "Original model", "Analysis"]
logger = setup_global_logger(
    script_name=script_base,
    cwd=LOG_FILES,
    log_level='INFO',
    headers=LOG_HEADER,
    enforce_standard_headers=False,
    log_initialization=False,
    log_suffix='.csv'
)
LOG_OUTPUT = LOG_FILES / f"a_{script_base}.csv"


def collect_stats(embeddings):
    if embeddings is None:
        return None
    norms = np.linalg.norm(embeddings, axis=1)
    similarity = None
    if not np.isnan(embeddings).any() and not np.allclose(norms, 0):
        normalized = embeddings / norms[:, np.newaxis]
        similarity = np.dot(normalized, normalized.T)
    return {
        "shape": embeddings.shape,
        "dtype": str(embeddings.dtype),
        "min": float(np.min(embeddings)),
        "max": float(np.max(embeddings)),
        "mean": float(np.mean(embeddings)),
        "std": float(np.std(embeddings)),
        "has_nan": bool(np.isnan(embeddings).any()),
        "has_inf": bool(np.isinf(embeddings).any()),
        "all_zero": bool(np.allclose(embeddings, 0)),
        "norm_min": float(np.min(norms)) if norms.size else 0.0,
        "norm_max": float(np.max(norms)) if norms.size else 0.0,
        "norms": norms,
        "similarity": similarity,
    }


def format_ndarray(arr):
    if arr is None:
        return "n/a"
    np_arr = np.array(arr)
    if np_arr.ndim == 1:
        return '[' + ' '.join(f"{val:.8f}" for val in np_arr) + ']'
    if np_arr.ndim == 2:
        rows = ['[' + ' '.join(f"{val:.8f}" for val in row) + ']' for row in np_arr]
        return '[' + ','.join(rows) + ']'
    return np.array2string(np_arr, precision=8, separator=',').replace('\n', '')


def format_value(value):
    if value is None:
        return "n/a"
    if isinstance(value, np.ndarray):
        return format_ndarray(value)
    if isinstance(value, (list, tuple)):
        return str(tuple(value)) if isinstance(value, tuple) else str(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _band(norms):
    if norms is None:
        return None
    return float(np.min(norms)), float(np.max(norms))


def _similarity_band(matrix):
    if matrix is None:
        return None
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    values = matrix[mask]
    if values.size == 0:
        return None
    return float(np.min(values)), float(np.max(values))


def note_shape(t_shape, o_shape):
    if t_shape and o_shape:
        if t_shape == o_shape:
            return "Same embedding dimension; no projection change."
        return ("Different embedding dimension; expected if tuned model uses a different "
                "pooling/projection head. Not directly comparable by dimension.")
    if t_shape and not o_shape:
        return "Original model shape missing; cannot compare dimensions."
    if o_shape and not t_shape:
        return "Tuned model shape missing; cannot compare dimensions."
    return "Shape comparison unavailable."


def note_dtype(t_dtype, o_dtype):
    if t_dtype and o_dtype:
        if t_dtype == o_dtype:
            return "Same dtype; no precision change."
        return f"Different dtypes (tuned={t_dtype}, original={o_dtype}); downstream ops must handle cast."
    return "Dtype comparison unavailable."


def note_nan(t_has, o_has):
    if t_has is None or o_has is None:
        return "NaN check unavailable."
    if not t_has and not o_has:
        return "Both outputs numerically valid (no NaNs)."
    if t_has and o_has:
        return "Both models emit NaNs; investigate tokenizer/data pipeline."
    if t_has:
        return "Tuned model emits NaNs while original stays clean; lower LR or fix data."
    return "Original model emits NaNs while tuned stays clean; baseline may be corrupted."


def note_inf(t_has, o_has):
    if t_has is None or o_has is None:
        return "Inf check unavailable."
    if not t_has and not o_has:
        return "Both outputs numerically valid (no Infs)."
    if t_has and o_has:
        return "Both models produce Inf values; investigate exploding activations."
    if t_has:
        return "Tuned model produces Infs while original does not; investigate training stability."
    return "Original model produces Infs while tuned does not; baseline appears unstable."


def note_all_zero(t_zero, o_zero):
    if t_zero is None or o_zero is None:
        return "Zero-vector check unavailable."
    if not t_zero and not o_zero:
        return "Both models produce non-degenerate embeddings."
    if t_zero and o_zero:
        return "Both models collapsed to zero vectors; invalid outputs."
    if t_zero:
        return "Tuned model collapsed to zero vectors; discard this run."
    return "Original model collapsed to zero vectors; baseline broken."


def note_min(t_val, o_val):
    if t_val is None or o_val is None:
        return "Minimum comparison unavailable."
    if abs(t_val - o_val) < 1e-6:
        return "Minimum values nearly identical."
    if t_val < o_val:
        return "Tuned model has a slightly wider negative range; not inherently better/worse."
    return "Original model has a slightly wider negative range; not inherently better/worse."


def note_max(t_val, o_val):
    if t_val is None or o_val is None:
        return "Maximum comparison unavailable."
    if abs(t_val - o_val) < 1e-6:
        return "Maximum values nearly identical."
    if t_val > o_val:
        return "Tuned model has a slightly wider positive range; scale differences often stem from projection heads."
    return "Original model has a slightly wider positive range; scale differences often stem from projection heads."


def note_mean(t_val, o_val):
    if t_val is None or o_val is None:
        return "Mean comparison unavailable."
    t_abs, o_abs = abs(t_val), abs(o_val)
    if t_abs < 1e-3 and o_abs < 1e-3:
        return "Both means near zero; indicates centered distribution, nothing alarming."
    if t_abs < o_abs:
        return "Tuned mean is closer to zero than original; slightly more centered."
    if o_abs < t_abs:
        return "Original mean is closer to zero than tuned; check for drift if this grows."
    return "Means differ but remain small; monitor if drift increases."


def note_std(t_val, o_val):
    if t_val is None or o_val is None:
        return "Std comparison unavailable."
    diff = t_val - o_val
    if abs(diff) < 1e-4:
        return "Variance nearly identical between models."
    if diff > 0:
        return ("Tuned model has higher variance (std {:.6f} vs {:.6f}). Can increase separability, "
                "but validate with retrieval metrics.").format(t_val, o_val)
    return ("Original model has higher variance (std {:.6f} vs {:.6f}). Can increase separability, "
            "but validate with retrieval metrics.").format(o_val, t_val)


def note_norms(t_stats, o_stats):
    t_band = _band(t_stats.get('norms') if t_stats else None)
    o_band = _band(o_stats.get('norms') if o_stats else None)
    if t_band and o_band:
        if max(abs(t_band[0] - 1.0), abs(t_band[1] - 1.0), abs(o_band[0] - 1.0), abs(o_band[1] - 1.0)) < 0.05:
            return "Both are ~unit-normalized; consistent with cosine-sim embedding usage."
        return (f"Norm bands differ (tuned {t_band[0]:.4f}-{t_band[1]:.4f}, "
                f"original {o_band[0]:.4f}-{o_band[1]:.4f}); ensure downstream normalization.")
    if t_band and not o_band:
        return "Original norms unavailable; tuned norms reported only."
    if o_band and not t_band:
        return "Tuned norms unavailable; original norms reported only."
    return "Norm comparison unavailable."


def note_similarity(t_matrix, o_matrix):
    t_band = _similarity_band(t_matrix)
    o_band = _similarity_band(o_matrix)
    if t_band and o_band:
        return (f"Pairwise similarities are in the same band (tuned ~{t_band[0]:.2f}-{t_band[1]:.2f}, "
                f"original ~{o_band[0]:.2f}-{o_band[1]:.2f}). No obvious qualitative win; need retrieval metrics.")
    if t_band and not o_band:
        return f"Tuned similarities span ~{t_band[0]:.2f}-{t_band[1]:.2f}; original matrix unavailable."
    if o_band and not t_band:
        return f"Original similarities span ~{o_band[0]:.2f}-{o_band[1]:.2f}; tuned matrix unavailable."
    return "Similarity matrix unavailable (NaNs or zero vectors prevented computation)."


def build_rows(t_stats, o_stats):
    rows = []

    def get(stats, key):
        return None if stats is None else stats.get(key)

    def add(item, tuned_val, orig_val, note):
        rows.append((item, format_value(tuned_val), format_value(orig_val), note))

    add("Embedding shape", get(t_stats, 'shape'), get(o_stats, 'shape'), note_shape(get(t_stats, 'shape'), get(o_stats, 'shape')))
    add("Embedding dtype", get(t_stats, 'dtype'), get(o_stats, 'dtype'), note_dtype(get(t_stats, 'dtype'), get(o_stats, 'dtype')))
    add("Contains NaN", get(t_stats, 'has_nan'), get(o_stats, 'has_nan'), note_nan(get(t_stats, 'has_nan'), get(o_stats, 'has_nan')))
    add("Contains Inf", get(t_stats, 'has_inf'), get(o_stats, 'has_inf'), note_inf(get(t_stats, 'has_inf'), get(o_stats, 'has_inf')))
    add("All zeros", get(t_stats, 'all_zero'), get(o_stats, 'all_zero'), note_all_zero(get(t_stats, 'all_zero'), get(o_stats, 'all_zero')))
    add("Min value", get(t_stats, 'min'), get(o_stats, 'min'), note_min(get(t_stats, 'min'), get(o_stats, 'min')))
    add("Max value", get(t_stats, 'max'), get(o_stats, 'max'), note_max(get(t_stats, 'max'), get(o_stats, 'max')))
    add("Mean", get(t_stats, 'mean'), get(o_stats, 'mean'), note_mean(get(t_stats, 'mean'), get(o_stats, 'mean')))
    add("Std", get(t_stats, 'std'), get(o_stats, 'std'), note_std(get(t_stats, 'std'), get(o_stats, 'std')))
    add("Embedding norms", get(t_stats, 'norms'), get(o_stats, 'norms'), note_norms(t_stats, o_stats))
    add("Manual similarity matrix", get(t_stats, 'similarity'), get(o_stats, 'similarity'), note_similarity(get(t_stats, 'similarity'), get(o_stats, 'similarity')))
    return rows


def classify_stats(stats):
    if stats is None:
        return "n/a", "not evaluated"
    if stats['has_nan'] or stats['has_inf']:
        return "poor", "contains invalid values (NaN/Inf detected)"
    if stats['all_zero'] or stats['norm_max'] == 0.0:
        return "poor", "collapsed to zero vectors"
    if stats['std'] > 1e-2 and stats['norm_min'] > 0.05:
        return "great", f"shows healthy spread (std={stats['std']:.4f}, norms {stats['norm_min']:.4f}-{stats['norm_max']:.4f})"
    return "good", f"remains stable (std={stats['std']:.4f}, norms {stats['norm_min']:.4f}-{stats['norm_max']:.4f})"


def compose_analysis(tuned_rating, tuned_note, orig_rating, orig_note):
    overall = tuned_rating if tuned_rating != "n/a" else orig_rating
    if overall == "n/a":
        overall = "n/a"
    parts = [f"{overall}: Tuned model {tuned_note}"]
    if orig_note:
        parts.append(f"Original model {orig_note}")
    return '. '.join(parts)


def format_detail_block(stats, norms=None, similarity_matrix=None):
    if stats is None:
        return "n/a"
    lines = [
        f"Embedding shape: {stats['shape']}",
        f"Embedding dtype: {stats['dtype']}",
        f"Contains NaN: {stats['has_nan']}",
        f"Contains Inf: {stats['has_inf']}",
        f"All zeros: {stats['all_zero']}",
        f"Min value: {stats['min']}",
        f"Max value: {stats['max']}",
        f"Mean: {stats['mean']}",
        f"Std: {stats['std']}",
    ]
    if norms is not None:
        lines.append(f"Embedding norms: {np.array2string(norms, precision=8)}")
    if similarity_matrix is not None:
        lines.append("Manual similarity matrix:")
        lines.append(np.array2string(similarity_matrix, precision=8))
    return "\n".join(lines)


def main():
    model_path = OUTPUT_MODEL_PATH
    print(f"Loading model from: {model_path}")

    try:
        model = SentenceTransformer(str(model_path))
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return

    test_texts = [
        "Function Summary",
        "Object Reference",
        "CMS Connector",
    ]

    print(f"\nTesting with {len(test_texts)} simple texts...")
    embeddings = model.encode(test_texts)
    tuned_stats = collect_stats(embeddings)
    print(f"Embedding shape: {tuned_stats['shape']}")
    print(f"Embedding dtype: {tuned_stats['dtype']}")
    print(f"Contains NaN: {tuned_stats['has_nan']}")
    print(f"Contains Inf: {tuned_stats['has_inf']}")
    print(f"All zeros: {tuned_stats['all_zero']}")
    print(f"Min value: {tuned_stats['min']}")
    print(f"Max value: {tuned_stats['max']}")
    print(f"Mean: {tuned_stats['mean']}")
    print(f"Std: {tuned_stats['std']}")
    norms = tuned_stats['norms']
    print(f"Embedding norms: {norms}")

    if tuned_stats['similarity'] is not None:
        print(f"Manual similarity matrix:\n{tuned_stats['similarity']}")
    else:
        print("Cannot compute similarity - embeddings contain NaN or are zero-magnitude")

    print("\n" + "=" * 50)
    print("Testing original Qwen model for comparison...")
    original_embeddings = None
    original_stats = None
    try:
        original_model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")
        original_embeddings = original_model.encode(test_texts)
        original_stats = collect_stats(original_embeddings)
        print(f"Original embedding shape: {original_stats['shape']}")
        print(f"Original contains NaN: {original_stats['has_nan']}")
        print(f"Original contains Inf: {original_stats['has_inf']}")
        print(f"Original all zeros: {original_stats['all_zero']}")
        print(f"Original min value: {original_stats['min']}")
        print(f"Original max value: {original_stats['max']}")
        print(f"Original mean: {original_stats['mean']}")
        print(f"Original std: {original_stats['std']}")
        print(f"Original embedding norms: {original_stats['norms']}")
        if original_stats['similarity'] is not None:
            print(f"Original similarity matrix:\n{original_stats['similarity']}")
        else:
            print("Original similarity matrix unavailable (NaNs or zero vectors)")
    except Exception as e:
        print(f"Could not load original model: {e}")
        original_stats = None

    rows = build_rows(tuned_stats, original_stats)
    for item, tuned_val, orig_val, note in rows:
        logger.info(
            item,
            extra={
                "Item": item,
                "Tuned model": tuned_val,
                "Original model": orig_val,
                "Analysis": note,
            },
        )

    print(f"\nCSV summary saved to: {LOG_OUTPUT}")


if __name__ == "__main__":
    main()