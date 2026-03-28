#!/usr/bin/env python3
"""
PDF, XML and Text Ingestion for Legal Documents
Parses PDFs, XML files, and text files, extracts legal metadata, and chunks text for RAG.
"""

import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional


def parse_text_file(filepath: Path) -> dict:
    """
    Parse a plain text file and return the same structure as parse_legal_pdf.

    Returns:
        {
            "full_text": str,
            "pages": [{"page_num": 1, "text": str, "bbox": {}}],
            "metadata": {
                "page_count": 1,
                "file_size": int,
                "created": datetime,
                "modified": datetime
            }
        }
    """
    text = filepath.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    return {
        "full_text": text,
        "pages": [{"page_num": 1, "text": text, "bbox": {}}],
        "metadata": {
            "page_count": 1,
            "file_size": filepath.stat().st_size,
            "created": datetime.fromtimestamp(filepath.stat().st_ctime),
            "modified": datetime.fromtimestamp(filepath.stat().st_mtime),
        },
    }


def parse_legal_xml(filepath: Path) -> dict:
    """
    Parse AustLII XML legal case file using BeautifulSoup.
    BeautifulSoup handles malformed XML better than ElementTree.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "BeautifulSoup4 not installed. Run: pip install beautifulsoup4"
        )

    content = filepath.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(content, "html.parser")

    case_name = soup.find("name")
    case_name = case_name.get_text(strip=True) if case_name else None

    austlii_elem = soup.find("AustLII")
    austlii_url = austlii_elem.get_text(strip=True) if austlii_elem else None

    catchphrases = []
    for cp in soup.find_all("catchphrase"):
        text = cp.get_text(strip=True)
        if text:
            catchphrases.append(text)

    full_text_parts = []
    for sentence in soup.find_all("sentence"):
        text = sentence.get_text(strip=True)
        if text:
            full_text_parts.append(text)

    full_text = "\n\n".join(full_text_parts)
    pages = [{"page_num": 1, "text": full_text, "bbox": {}}]

    return {
        "full_text": full_text,
        "pages": pages,
        "metadata": {
            "page_count": 1,
            "file_size": filepath.stat().st_size,
            "created": datetime.fromtimestamp(filepath.stat().st_ctime),
            "modified": datetime.fromtimestamp(filepath.stat().st_mtime),
            "case_name": case_name,
            "catchphrases": catchphrases,
            "austlii_url": austlii_url,
        },
    }


def parse_legal_pdf(filepath: Path) -> dict:
    """
    Parse PDF and extract text with page-level metadata.

    Returns:
        {
            "full_text": str,
            "pages": [
                {"page_num": int, "text": str, "bbox": dict}
            ],
            "metadata": {
                "page_count": int,
                "file_size": int,
                "created": datetime,
                "modified": datetime
            }
        }
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

    doc = fitz.open(filepath)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        pages.append(
            {
                "page_num": page_num,
                "text": text,
                "bbox": {
                    "x0": page.rect.x0,
                    "y0": page.rect.y0,
                    "x1": page.rect.x1,
                    "y1": page.rect.y1,
                },
            }
        )

    return {
        "full_text": "\n\n".join(p["text"] for p in pages),
        "pages": pages,
        "metadata": {
            "page_count": len(doc),
            "file_size": filepath.stat().st_size,
            "created": datetime.fromtimestamp(filepath.stat().st_ctime),
            "modified": datetime.fromtimestamp(filepath.stat().st_mtime),
        },
    }


def extract_legal_metadata(text: str, filename: str) -> dict:
    """
    Extract legal-specific metadata from document text.

    Returns:
        {
            "case_name": str | None,
            "citation": str | None,
            "court": str | None,
            "jurisdiction": str | None,
            "year": int | None,
            "judge": str | None
        }
    """
    metadata = {
        "case_name": "",
        "citation": "",
        "court": "",
        "jurisdiction": "",
        "year": None,
        "judge": "",
    }

    year_match = re.search(r"\b(19|20)\d{2}\b", filename)
    if year_match:
        metadata["year"] = int(year_match.group(0))

    citation_match = re.search(r"\[(\d{4})\]\s+([A-Z]+)\s+(\d+)", text[:2000])
    if citation_match:
        metadata["year"] = int(citation_match.group(1))
        metadata["court"] = citation_match.group(2)
        metadata["citation"] = citation_match.group(0)

    case_name_match = re.search(
        r"^([A-Z][a-zA-Z\s]+v\.?\s+[A-Z][a-zA-Z]+)", text[:500], re.MULTILINE
    )
    if case_name_match:
        metadata["case_name"] = case_name_match.group(1).strip()

    judge_match = re.search(
        r"(Justice|Judge|Chief Justice)\s+([A-Z][a-zA-Z]+)", text[:2000]
    )
    if judge_match:
        metadata["judge"] = f"{judge_match.group(1)} {judge_match.group(2)}"

    jurisdiction_match = re.search(
        r"(Supreme Court|District Court|County Court|Federal Court|Family Court)",
        text[:2000],
    )
    if jurisdiction_match:
        jurisdiction = jurisdiction_match.group(1)
        if "NSW" in text[:500]:
            metadata["jurisdiction"] = f"NSW - {jurisdiction}"
        elif "VIC" in text[:500]:
            metadata["jurisdiction"] = f"VIC - {jurisdiction}"
        elif "QLD" in text[:500]:
            metadata["jurisdiction"] = f"QLD - {jurisdiction}"
        else:
            metadata["jurisdiction"] = jurisdiction

    return metadata


def extract_xml_metadata(parsed: dict, filename: str) -> dict:
    """
    Extract legal metadata from parsed XML document.

    Returns:
        {
            "case_name": str,
            "citation": str,
            "court": str,
            "year": int,
            "judge": str,
            "case_type": str,
            "catchphrases": list[str]
        }
    """
    metadata = {
        "case_name": "",
        "citation": "",
        "court": "",
        "jurisdiction": "",
        "year": None,
        "judge": "",
        "case_type": "",
        "catchphrases": [],
    }

    xml_meta = parsed.get("metadata", {})

    if xml_meta.get("case_name"):
        metadata["case_name"] = xml_meta["case_name"]

        citation_match = re.search(
            r"\[(\d{4})\]\s+([A-Z]+)\s+(\d+)", xml_meta["case_name"]
        )
        if citation_match:
            metadata["year"] = int(citation_match.group(1))
            metadata["court"] = citation_match.group(2)
            metadata["citation"] = (
                f"[{citation_match.group(1)}] {citation_match.group(2)} {citation_match.group(3)}"
            )

    if xml_meta.get("catchphrases"):
        metadata["catchphrases"] = xml_meta["catchphrases"]
        cp_text = " ".join(xml_meta["catchphrases"]).lower()
        if "copyright" in cp_text:
            metadata["case_type"] = "copyright"
        elif "contract" in cp_text:
            metadata["case_type"] = "contract"
        elif "negligence" in cp_text or "damages" in cp_text:
            metadata["case_type"] = "tort"
        elif "employment" in cp_text:
            metadata["case_type"] = "employment"

    if not metadata["year"]:
        year_match = re.search(r"\b(19|20)\d{2}\b", filename)
        if year_match:
            metadata["year"] = int(year_match.group(0))

    return metadata


def chunk_pdf(
    pages: list[dict],
    chunk_size: int = 400,
    overlap: int = 80,
    doc_metadata: Optional[dict] = None,
) -> list[dict]:
    """
    Chunk PDF text with sliding window and paragraph preservation.

    Args:
        pages: List of page dicts from parse_legal_pdf
        chunk_size: Target words per chunk
        overlap: Overlap words between chunks
        doc_metadata: Document-level metadata to attach to all chunks

    Returns:
        List of chunk dicts with text + metadata
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        page_num = page["page_num"]
        text = page["text"]

        paragraphs = re.split(r"\n\n+", text)

        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            words = para.split()
            para_length = len(words)

            if current_length + para_length > chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    {
                        "text": chunk_text,
                        "chunk_id": chunk_id,
                        "page_num": page_num,
                        "word_count": len(current_chunk),
                        **(doc_metadata or {}),
                    }
                )
                chunk_id += 1

                overlap_words = current_chunk[-overlap:] if overlap > 0 else []
                current_chunk = overlap_words
                current_length = len(overlap_words)

            current_chunk.extend(words)
            current_length += para_length

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                {
                    "text": chunk_text,
                    "chunk_id": chunk_id,
                    "page_num": page_num,
                    "word_count": len(current_chunk),
                    **(doc_metadata or {}),
                }
            )
            chunk_id += 1

    return chunks


def generate_test_pdfs(output_dir: Path, count: int = 3) -> list[Path]:
    """
    Generate synthetic legal PDFs for testing.

    Args:
        output_dir: Directory to save PDFs
        count: Number of PDFs to generate

    Returns:
        List of generated file paths
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        raise ImportError("reportlab not installed. Run: pip install reportlab")

    output_dir.mkdir(parents=True, exist_ok=True)

    test_cases = [
        {
            "filename": "smith_v_jones_2023.pdf",
            "case_name": "Smith v Jones",
            "year": 2023,
            "court": "NSWSC",
            "citation": "[2023] NSWSC 145",
            "judge": "Justice Smith",
            "content": """
SMITH v JONES

[2023] NSWSC 145

Supreme Court of New South Wales

Justice Smith

CASE SUMMARY

This is an action for breach of contract arising from the failure to deliver
goods under a commercial supply agreement. The plaintiff, Mr. Smith, entered
into a written contract with the defendant, Jones Pty Ltd, for the supply
of office equipment worth $150,000.

The defendant failed to deliver the goods by the agreed date of 15 March 2023.
The plaintiff contends that this delay caused significant financial loss.

ISSUES FOR DETERMINATION

1. Whether a binding contract existed between the parties
2. Whether the defendant breached the contract
3. What damages, if any, is the plaintiff entitled to

HELD

The Court finds in favour of the plaintiff. There was a clear and unambiguous
contract between the parties. The defendant's failure to deliver constituted
a material breach.

Judgment is entered for the plaintiff in the sum of $150,000 plus damages
for consequential losses of $25,000.

ORDERS

1. Judgment for the plaintiff
2. Damages assessed at $175,000
3. Costs on the ordinary scale
            """,
        },
        {
            "filename": "doe_v_acme_2024.pdf",
            "case_name": "Doe v Acme Corp",
            "year": 2024,
            "court": "FCA",
            "citation": "[2024] FCA 234",
            "judge": "Justice Brown",
            "content": """
DOE v ACME CORPORATION

[2024] FCA 234

Federal Court of Australia

Justice Brown

CASE SUMMARY

This matter concerns a claim for wrongful dismissal under the Fair Work Act.
The applicant, Ms. Doe, was employed by Acme Corporation as a senior manager
for a period of five years.

The applicant was terminated without notice or proper procedural fairness.
The respondent claims the termination was for serious misconduct.

ISSUES FOR DETERMINATION

1. Whether the termination was valid
2. Whether procedural fairness was afforded
3. Whether compensation is payable

FINDINGS

The Court finds that the termination was not justified. The respondent failed
to follow proper procedures and did not provide the applicant with an opportunity
to respond to the allegations.

The applicant's claim for compensation is upheld.

ORDERS

1. Compensation awarded: 12 months' salary ($180,000)
2. Reinstatement offered as alternative
3. Costs reserved
            """,
        },
        {
            "filename": "brown_v_green_2022.pdf",
            "case_name": "Brown v Green",
            "year": 2022,
            "court": "VSC",
            "citation": "[2022] VSC 89",
            "judge": "Justice White",
            "content": """
BROWN v GREEN

[2022] VSC 89

Supreme Court of Victoria

Justice White

CASE SUMMARY

This is a personal injury claim arising from a motor vehicle accident.
The plaintiff, Mr. Brown, was struck by a vehicle driven by the defendant,
Mr. Green, at a busy intersection.

The plaintiff suffered serious injuries including a fractured pelvis and
traumatic brain injury. Liability is admitted by the defendant.

ISSUES FOR DETERMINATION

1. Assessment of damages
2. Future care requirements
3. Loss of earning capacity

ASSESSMENT

The Court accepts the medical evidence tendered by the plaintiff. The injuries
have resulted in permanent disability and significant ongoing care needs.

Damages are assessed as follows:

- Past economic loss: $85,000
- Future economic loss: $250,000
- Future care: $150,000
- General damages: $120,000

TOTAL: $605,000

ORDERS

Judgment for the plaintiff in the sum of $605,000
            """,
        },
    ]

    generated_files = []
    for i, case in enumerate(test_cases[:count]):
        filepath = output_dir / case["filename"]

        c = canvas.Canvas(str(filepath), pagesize=letter)
        width, height = letter

        y = height - 50
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, case["case_name"])

        y -= 30
        c.setFont("Helvetica", 12)
        c.drawString(50, y, case["citation"])

        y -= 20
        c.drawString(
            50,
            y,
            f"Supreme Court" if case["court"] in ["NSWSC", "VSC"] else "Federal Court",
        )

        y -= 20
        c.drawString(50, y, case["judge"])

        y -= 40
        c.setFont("Helvetica", 10)

        for line in case["content"].strip().split("\n"):
            line = line.strip()
            if not line:
                y -= 15
                continue
            if y < 50:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 10)
            c.drawString(50, y, line)
            y -= 15

        c.save()
        generated_files.append(filepath)
        print(f"Generated: {filepath}")

    return generated_files


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--generate-test-pdfs":
        output_dir = Path(__file__).parent.parent / "data" / "legal-docs"
        files = generate_test_pdfs(output_dir)
        print(f"\nGenerated {len(files)} test PDFs in {output_dir}")
    else:
        print("Legal PDF Ingestion Module")
        print("Usage:")
        print("  python -m pdf_ingest --generate-test-pdfs")
