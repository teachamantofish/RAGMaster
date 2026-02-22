"""
Simple PDF page trimmer using PyMuPDF (fitz).

Purpose: Remove pages from the front, middle, or end of a PDF. No TOC work, no CLI.
Configure the constants below and run the script, or import and call trim_pdf().

Notes:
- Page numbers in REMOVE_RANGES are 1-based inclusive (e.g., (5, 10) removes pages 5..10).
- When OUT_PDF is None, the input file is modified in place.
- We copy kept pages to a new document to avoid in-place deletion pitfalls and file locks.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Sequence, Tuple
import fitz  # PyMuPDF

# =========================
# Configuration (edit these)
# =========================
SRC_PDF: Path = Path(r"C:\GIT\ai_data\extendscript\firsthalf.pdf")
OUT_PDF: Path | None = Path(r"C:\GIT\ai_data\extendscript\firsthalf_trimmed.pdf")
REMOVE_FRONT: int = 0
REMOVE_BACK: int = 500
REMOVE_RANGES: List[Tuple[int, int]] = []

def trim_pdf(
    src_pdf: Path | str,
    remove_front: int = 0,
    remove_back: int = 0,
    remove_ranges: Sequence[Tuple[int, int]] | None = None,
    out_pdf: Path | str | None = None,
) -> Path:
    """Trim pages from a PDF and save the result.

    Inputs
    - src_pdf: path to the source PDF.
    - remove_front: number of pages to remove from the beginning (0-based positions 0..remove_front-1).
    - remove_back: number of pages to remove from the end.
    - remove_ranges: 1-based inclusive page ranges to remove, e.g., [(5, 10), (23, 23)].
    - out_pdf: destination path. If None, overwrite src_pdf (in-place).

    Returns
    - Path to the written PDF.

    Raises
    - ValueError if resulting document would be empty.
    """

    src_path = Path(src_pdf)
    dest_path = Path(out_pdf) if out_pdf is not None else src_path

    doc = fitz.open(str(src_path))
    out = None
    try:
        n = doc.page_count

        # Coerce to non-negative counts; treat negative inputs as their magnitude (e.g., 0-89 -> 89)
        rf = max(0, abs(int(remove_front or 0)))
        rb = max(0, abs(int(remove_back or 0)))

        # Compute primary keep window [start, end) in 0-based indexing
        start = min(rf, n)
        end = max(start, n - rb)

        # Normalize user-provided ranges (1-based inclusive) into a list of (lo, hi) ints
        drops: List[Tuple[int, int]] = []
        if remove_ranges:
            for lo, hi in remove_ranges:
                lo_i = int(lo)
                hi_i = int(hi)
                if lo_i > hi_i:
                    lo_i, hi_i = hi_i, lo_i
                # clamp to valid 1..n
                lo_i = max(1, min(n, lo_i))
                hi_i = max(1, min(n, hi_i))
                drops.append((lo_i, hi_i))

        # Debug: show effective parameters
        print(
            f"[trim] src={src_path} out={dest_path} remove_front={rf} remove_back={rb} remove_ranges={drops}"
        )

        def is_dropped(page_one_based: int) -> bool:
            return any(lo <= page_one_based <= hi for (lo, hi) in drops)

        # Build keep set in 0-based indices, excluding any pages in drop ranges
        keep_indices = [p for p in range(start, end) if not is_dropped(p + 1)]

        if not keep_indices:
            raise ValueError(
                "All pages would be removed — adjust remove_front/remove_back/remove_ranges."
            )

        out = fitz.open()
        try:
            for p in keep_indices:
                out.insert_pdf(doc, from_page=p, to_page=p)
        finally:
            # Close source before saving to the same path on Windows (file locks)
            doc.close()

        # Save
        out.save(dest_path, garbage=4, deflate=True, clean=True)
        out.close()

        print(
            f"[trim] input={n} keep={len(keep_indices)} (front={rf}, back={rb}, middle={len(drops)} ranges) -> {dest_path}"
        )
        return dest_path

    except Exception:
        # Ensure source is closed on exception
        try:
            doc.close()
        except Exception:
            pass
        try:
            if out is not None:
                out.close()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    # Run with the constants defined above
    trim_pdf(
        src_pdf=SRC_PDF,
        remove_front=REMOVE_FRONT,
        remove_back=REMOVE_BACK,
        remove_ranges=REMOVE_RANGES,
        out_pdf=OUT_PDF,
    )
