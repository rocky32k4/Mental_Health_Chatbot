"""
ingestion.py
------------
Phase 2 : Data Preparation Pipeline

Does 3 things in sequence:
  1. Extract raw text from PDF files
  2. Clean the extracted text
  3. Chunk the cleaned text and save to JSON

Run:
    python src/ingestion.py
"""

import os
import re
import json
import pdfplumber
from pathlib import Path


# ─────────────────────────────────────────────
# PATHS & SETTINGS
# ─────────────────────────────────────────────

RAW_DATA_DIR  = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTPUT_FILE   = PROCESSED_DIR / "chunks.json"

PDF_FILES = {
    "motivational_interviewing": "Motivational_Interviewing_for_Clinical_Practice_Petros_Levounis.pdf",
    "suicide_prevention":        "Suicide_prvention_handbook.pdf",
}

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50


# ─────────────────────────────────────────────
# STEP 1 — EXTRACT
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file, page by page."""
    print(f"  Extracting : {pdf_path.name}")
    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"  Pages      : {total}")

        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                full_text.append(text)
            if (i + 1) % 50 == 0:
                print(f"  Progress   : {i + 1}/{total} pages")

    combined = "\n".join(full_text)
    print(f"  Characters : {len(combined):,}")
    return combined


# ─────────────────────────────────────────────
# STEP 2 — CLEAN
# ─────────────────────────────────────────────

def is_junk_chunk(text: str) -> bool:
    """
    Returns True if a chunk is not useful content.
    Filters out:
      - Table of contents  (lots of dots like ". . . . .")
      - Index pages        (short lines with page numbers)
      - Nearly empty chunks
    """
    # Too short to be useful
    if len(text.strip()) < 80:
        return True

    # Table of contents — contains many dot sequences
    dot_sequences = len(re.findall(r'\.{3,}', text))
    if dot_sequences > 3:
        return True

    # Mostly dots and numbers (TOC pattern)
    non_alpha = len(re.findall(r'[^a-zA-Z\s]', text))
    total     = len(text)
    if total > 0 and (non_alpha / total) > 0.4:
        return True

    return False


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text.

    Fixes:
      - Hyphenated line breaks   (re-  joined -> rejoined)
      - Standalone page numbers
      - Extra blank lines
      - Multiple spaces
      - Non-ASCII characters
    """
    # Fix hyphenated line breaks
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Remove standalone page numbers
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Replace tabs
    text = re.sub(r"\t+", " ", text)

    # Remove multiple spaces
    text = re.sub(r" {2,}", " ", text)

    # Remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    return text.strip()


# ─────────────────────────────────────────────
# STEP 3 — CHUNK
# ─────────────────────────────────────────────

def chunk_text(text: str, source: str) -> list[dict]:
    """
    Split cleaned text into overlapping chunks.
    Skips junk chunks (TOC, index pages, etc.)

    Each chunk is a dict:
      - chunk_id   : unique ID e.g. "motivational_interviewing_0042"
      - source     : which book
      - text       : actual content
      - char_start : start position in original text
      - char_end   : end position in original text
    """
    chunks      = []
    start       = 0
    chunk_index = 0
    skipped     = 0

    while start < len(text):
        end     = start + CHUNK_SIZE
        content = text[start:end]

        if is_junk_chunk(content):
            skipped += 1
        else:
            chunks.append({
                "chunk_id"  : f"{source}_{chunk_index:04d}",
                "source"    : source,
                "text"      : content.strip(),
                "char_start": start,
                "char_end"  : min(end, len(text)),
            })
            chunk_index += 1

        start += CHUNK_SIZE - CHUNK_OVERLAP

    print(f"  Chunks     : {len(chunks)} kept, {skipped} junk skipped")
    return chunks


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_ingestion() -> list[dict]:
    """
    Run the full ingestion pipeline for all PDFs.
    Saves all chunks to data/processed/chunks.json
    """
    print("=" * 55)
    print("  PHASE 2 — INGESTION PIPELINE")
    print("=" * 55)

    all_chunks = []

    for source_name, filename in PDF_FILES.items():
        pdf_path = RAW_DATA_DIR / filename

        if not pdf_path.exists():
            print(f"\nFile not found: {pdf_path}")
            print(f"Please place '{filename}' in the '{RAW_DATA_DIR}' folder.\n")
            continue

        print(f"\nProcessing : {source_name}")
        print("-" * 40)

        raw_text     = extract_text_from_pdf(pdf_path)
        cleaned_text = clean_text(raw_text)
        chunks       = chunk_text(cleaned_text, source=source_name)

        all_chunks.extend(chunks)

    # Save
    if all_chunks:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 55)
        print("  INGESTION SUMMARY")
        print("=" * 55)

        sources = {}
        for chunk in all_chunks:
            sources[chunk["source"]] = sources.get(chunk["source"], 0) + 1
        for source, count in sources.items():
            print(f"  {source:<40} {count} chunks")
        print(f"  {'TOTAL':<40} {len(all_chunks)} chunks")
        print(f"\n  Saved to: {OUTPUT_FILE}")
        print("=" * 55)
    else:
        print("\nNo chunks created. Check PDF files are in data/raw/")

    return all_chunks


def load_chunks() -> list[dict]:
    """Load saved chunks. Used by vectorstore.py."""
    if not OUTPUT_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found at {OUTPUT_FILE}. "
            "Run ingestion first: python src/ingestion.py"
        )
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {OUTPUT_FILE}")
    return chunks


if __name__ == "__main__":
    run_ingestion()
