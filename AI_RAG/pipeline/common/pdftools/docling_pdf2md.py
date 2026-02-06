import os
from anyio import Path
import torch
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"  # add this before importing docling/huggingface
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
# --- parallel split helpers (uses PyPDF2 only to split; Docling options untouched) ---
from concurrent.futures import ThreadPoolExecutor
from tempfile import mkstemp
from PyPDF2 import PdfReader, PdfWriter  
# -------------------------------------------------------------------------------------


# need the layout and tableformer models. These are automatically downloaded if not present
# dont know if they persist between runs
# add ongoing status indicator so I know its working
# install HF model locally

# disable pinned memory to avoid the errors
torch.backends.cuda.matmul.allow_tf32 = False  # optional tweak
# monkeypatch dataloader defaults
from torch.utils import data
old_init = data.DataLoader.__init__
def new_init(self, *args, **kwargs):
    kwargs["pin_memory"] = False
    old_init(self, *args, **kwargs)
data.DataLoader.__init__ = new_init

# Set conversion options and convert the document
# See https://docling-project.github.io/docling/reference/pipeline_options/#docling.datamodel.pipeline_options.PdfPipelineOptions
pipeline_options = PdfPipelineOptions(
    do_table_structure=True,
    do_ocr=False,
    generate_picture_images=False,                           # keep: skip picture images
    generate_page_images=False,                              # keep: skip page images
    # pdf_backend=PdfPipelineBackend.PYPDFIUM2,              # pypdfium2
    table_structure_options={"mode": TableFormerMode.FAST},  # Choices: ACCURATE or FAST
    num_threads=12,                                          # 
    page_batch_size=6,
    preserve_whitespace=True,
    text_normalization=True,  # still normalize prose (remove odd symbols as well)
    # do_code_enrichment=True  # See https://huggingface.co/ds4sd/CodeFormula: breaks too much.
)
# use more accurate TableFormer model
pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    },
)

# --- helpers to run two parallel jobs and merge output ---
def _split_pdf_two_halves(src_path: str):
    reader = PdfReader(src_path)
    n = len(reader.pages)
    first_end = max(0, (n // 2) - 1)
    second_start = first_end + 1

    def _write_range(start_idx: int, end_idx: int) -> str:
        writer = PdfWriter()
        for i in range(start_idx, end_idx + 1):
            writer.add_page(reader.pages[i])
        tmp_path = mkstemp(suffix=".pdf")[1]
        with open(tmp_path, "wb") as f:
            writer.write(f)
        return tmp_path

    part1 = _write_range(0, first_end)
    part2 = _write_range(second_start, n - 1)
    return part1, part2

def _convert_one(pdf_path: str, out_name: str) -> str:
    converted = doc_converter.convert(pdf_path)
    doc = getattr(converted, "document", converted)
    with open(out_name, "w", encoding="utf-8") as f:
        f.write(doc.export_to_markdown())
    return out_name

def convert_in_two_jobs(src_pdf: str, job_cwd: str = None, csvrow_data: dict = None):
    # Caller provides job_cwd
    part1, part2 = _split_pdf_two_halves(src_pdf)
    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(_convert_one, part1, "temp1.md")
            f2 = ex.submit(_convert_one, part2, "temp2.md")
            md1 = f1.result()
            md2 = f2.result()

        # merged filename should be same as original but with .md extension
        base = os.path.splitext(os.path.basename(src_pdf))[0]
        merged_path = os.path.join(job_cwd, f"{base}.md")
        # Read both parts, merge into a single markdown string, then
        # merge page-level metadata into the global metadata and write a
        # single normalized front matter block followed by cleaned body.
        with open(md1, "r", encoding="utf-8") as a, open(md2, "r", encoding="utf-8") as b:
            merged_text = a.read() + "\n\n" + b.read()

        # lazy import so this module doesn't require common.* at import time
        from common.metadata_utils import merge_page_metadata
        # metadata must be present (provided by orchestrator). Use it directly.
        metadata = csvrow_data['input_csv_row']
        file_meta = dict(metadata)
        file_meta['CRAWL_URL'] = src_pdf
        merged_front, cleaned_body = merge_page_metadata(file_meta, merged_text)

        with open(merged_path, "w", encoding="utf-8") as out_f:
            out_f.write(merged_front + cleaned_body)
        print(f"Done. Output saved to {merged_path}")
        return merged_path
    finally:
        os.remove("temp1.md")
        os.remove("temp2.md")
# --------------------------------------------------------

if __name__ == "__main__":
    convert_in_two_jobs(source, job_cwd)
