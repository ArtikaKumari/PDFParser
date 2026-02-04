"""TOC (Table of Contents) detection, parsing, and index generation.

Scans a PDF's extracted pages to find a TOC/index page, parses its entries
to map document sections to page ranges, and provides a generated index
when no TOC exists in the document.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from pdf_classifier.extractor import PageContent
from pdf_classifier.taxonomy import DocumentType

logger = logging.getLogger(__name__)


@dataclass
class TOCEntry:
    """A single entry parsed from a table of contents."""

    label: str
    start_page: int
    end_page: int | None = None
    mapped_type: DocumentType | None = None


@dataclass
class DocumentIndex:
    """An index mapping page ranges to document types.

    Can be built from a parsed TOC or generated from classification results.
    """

    entries: list[TOCEntry] = field(default_factory=list)
    source: str = ""  # "toc" or "generated"
    toc_page_numbers: list[int] = field(default_factory=list)

    def get_type_for_page(self, page_number: int) -> DocumentType | None:
        """Look up the document type for a given page number."""
        for entry in self.entries:
            end = entry.end_page or entry.start_page
            if entry.start_page <= page_number <= end and entry.mapped_type:
                return entry.mapped_type
        return None

    def is_toc_page(self, page_number: int) -> bool:
        """Check if a page number is a TOC page itself."""
        return page_number in self.toc_page_numbers


# ---------------------------------------------------------------------------
# Label -> DocumentType mapping
# ---------------------------------------------------------------------------

_LABEL_TO_TYPE: list[tuple[list[str], DocumentType]] = [
    # Order matters: more specific patterns first
    (
        ["discharge summary", "discharge letter"],
        DocumentType.ED_DISCHARGE_SUMMARY,
    ),
    (
        ["operation report", "operative report", "surgical report"],
        DocumentType.ED_OPERATION_REPORT,
    ),
    (
        [
            "radiology", "imaging report", "x-ray", "xray", "ct scan",
            "mri report", "mri scan", "ultrasound report", "emg",
            "nerve conduction",
        ],
        DocumentType.RADIOLOGY,
    ),
    (
        ["outpatient", "out-patient", "clinic review", "clinic notes"],
        DocumentType.ED_OUTPATIENT_NOTES,
    ),
    (
        [
            "medicolegal", "medico-legal", "independent medical",
            "ime report", "impairment assessment",
        ],
        DocumentType.MEDICOLEGAL_REPORT,
    ),
    (
        ["medical panel", "panel report", "panel opinion"],
        DocumentType.MEDICAL_PANEL_REPORT,
    ),
    (
        ["affidavit", "statutory declaration"],
        DocumentType.AFFIDAVIT,
    ),
    (
        ["referral letter", "referral"],
        DocumentType.REFERRAL_LETTER,
    ),
    (
        [
            "physiotherapy", "physio notes", "physiotherapist",
            "occupational therapy", "psychology report", "psychologist",
            "allied health",
        ],
        DocumentType.CLINICAL_NOTES_ALLIED,
    ),
    (
        [
            "treater letter", "treating doctor", "gp letter",
            "general practitioner letter", "specialist letter",
        ],
        DocumentType.TREATER_LETTER_NON_ALLIED,
    ),
    (
        ["clinical notes", "clinical records", "medical notes", "consultation notes"],
        DocumentType.CLINICAL_NOTES_NON_ALLIED,
    ),
    (
        [
            "specialist", "orthopaedic", "orthopedic", "neurologist",
            "neurology", "cardiologist", "psychiatrist", "rheumatologist",
        ],
        DocumentType.SPECIALIST_CLINICAL_NOTES,
    ),
    (
        [
            "hospital", "emergency department", "ed presentation",
            "admission", "inpatient",
        ],
        DocumentType.ED_OTHER,
    ),
    (
        [
            "ambulance", "paramedic", "progress notes", "nursing notes",
            "pathology", "blood test", "laboratory",
        ],
        DocumentType.ED_OTHER,
    ),
]


def _map_label_to_type(label: str) -> DocumentType | None:
    """Map a TOC entry label to a DocumentType."""
    label_lower = label.lower()
    for keywords, doc_type in _LABEL_TO_TYPE:
        for kw in keywords:
            if kw in label_lower:
                return doc_type
    return None


# ---------------------------------------------------------------------------
# TOC detection
# ---------------------------------------------------------------------------

_TOC_HEADINGS: list[str] = [
    "table of contents",
    "contents",
    "index",
    "list of enclosures",
    "list of documents",
    "list of annexures",
    "list of appendices",
    "enclosure index",
    "document index",
    "paginated index",
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def is_toc_page(text: str) -> bool:
    """Detect if a page is a table of contents or index page."""
    normalized = _normalize(text)

    # Check for explicit TOC headings
    for heading in _TOC_HEADINGS:
        if heading in normalized:
            return True

    # Heuristic: dotted leaders with page numbers (e.g. "...... 12")
    dotted_refs = re.findall(r"\.{2,}\s*\d+", normalized)
    if len(dotted_refs) >= 3:
        return True

    # Heuristic: many "page N" / "pg N" references
    page_refs = re.findall(r"\b(?:page|pg|p)\s*\.?\s*\d+", normalized)
    if len(page_refs) >= 5:
        return True

    return False


def find_toc_pages(pages: list[PageContent]) -> list[PageContent]:
    """Find all TOC/index pages in the document.

    Args:
        pages: All extracted pages from the PDF.

    Returns:
        List of pages identified as TOC pages (usually 0 or 1).
    """
    toc_pages = []
    for page in pages:
        if page.text.strip() and is_toc_page(page.text):
            toc_pages.append(page)
    return toc_pages


# ---------------------------------------------------------------------------
# TOC parsing
# ---------------------------------------------------------------------------

# Patterns for parsing TOC entries like:
#   "Discharge Summary ........... 3"
#   "1. Radiology Report - pg 5"
#   "Enclosure 2: MRI Report ... page 8"
_TOC_ENTRY_PATTERNS: list[str] = [
    # "Label ........... 12" or "Label ... 12"
    r"(?P<label>[A-Za-z][^\n\.]{3,}?)\s*\.{2,}\s*(?P<page>\d+)",
    # "Label - page 12" or "Label - pg 12" or "Label - p 12"
    r"(?P<label>[A-Za-z][^\n-]{3,}?)\s*[-–]\s*(?:page|pg|p)\.?\s*(?P<page>\d+)",
    # "Label    12" (label followed by large whitespace then page number)
    r"(?P<label>[A-Za-z][^\n\d]{5,}?)\s{3,}(?P<page>\d+)\s*$",
    # "1. Label ... 12" or "Enclosure 1: Label ... 12"
    r"(?:(?:\d+[\.\):]|enclosure\s+\d+[:\.]?)\s*)(?P<label>[A-Za-z][^\n\.]{3,}?)\s*\.{2,}\s*(?P<page>\d+)",
]


def parse_toc_entries(toc_text: str) -> list[TOCEntry]:
    """Parse TOC text into structured entries with page numbers.

    Args:
        toc_text: Raw text from a TOC page.

    Returns:
        List of TOCEntry objects sorted by start_page.
    """
    entries: list[TOCEntry] = []
    seen_pages: set[int] = set()

    for line in toc_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        for pattern in _TOC_ENTRY_PATTERNS:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                label = match.group("label").strip()
                label = re.sub(r"^[\d\.\)\:]+\s*", "", label).strip()
                label = re.sub(r"[\.\s]+$", "", label).strip()

                if len(label) < 3:
                    continue

                page_num = int(match.group("page"))
                if page_num <= 0 or page_num in seen_pages:
                    continue

                mapped_type = _map_label_to_type(label)
                entries.append(TOCEntry(
                    label=label,
                    start_page=page_num,
                    mapped_type=mapped_type,
                ))
                seen_pages.add(page_num)
                break

    # Sort by start page
    entries.sort(key=lambda e: e.start_page)

    # Infer end pages: each entry runs until the next entry starts
    for i in range(len(entries) - 1):
        entries[i].end_page = entries[i + 1].start_page - 1

    # Last entry: leave end_page as None (handled by caller or set to total)
    return entries


def _set_last_entry_end_page(
    entries: list[TOCEntry], total_pages: int
) -> None:
    """Set the end_page of the last entry to the document's total pages."""
    if entries:
        if entries[-1].end_page is None:
            entries[-1].end_page = total_pages


# ---------------------------------------------------------------------------
# Build index from TOC
# ---------------------------------------------------------------------------


def build_index_from_toc(
    pages: list[PageContent], total_pages: int
) -> DocumentIndex | None:
    """Attempt to find and parse a TOC to build a document index.

    Args:
        pages: All extracted pages.
        total_pages: Total number of pages in the PDF.

    Returns:
        A DocumentIndex if a TOC was found and parsed, else None.
    """
    toc_pages = find_toc_pages(pages)

    if not toc_pages:
        logger.info("No TOC/index page found in document")
        return None

    if len(toc_pages) > 1:
        logger.warning(
            "Multiple TOC pages found (%d). Using the first one.",
            len(toc_pages),
        )

    toc_page = toc_pages[0]
    entries = parse_toc_entries(toc_page.text)

    if not entries:
        logger.info(
            "TOC page found (page %d) but could not parse any entries",
            toc_page.page_number,
        )
        return None

    _set_last_entry_end_page(entries, total_pages)

    mapped_count = sum(1 for e in entries if e.mapped_type is not None)
    logger.info(
        "TOC parsed: %d entries, %d mapped to document types (page %d)",
        len(entries),
        mapped_count,
        toc_page.page_number,
    )

    return DocumentIndex(
        entries=entries,
        source="toc",
        toc_page_numbers=[p.page_number for p in toc_pages],
    )


# ---------------------------------------------------------------------------
# Generate index from classification results (when no TOC exists)
# ---------------------------------------------------------------------------


def generate_index_from_classifications(
    page_types: list[tuple[int, DocumentType]],
) -> DocumentIndex:
    """Generate a synthetic index by grouping consecutive same-type pages.

    Args:
        page_types: List of (page_number, document_type) tuples, sorted
                    by page number.

    Returns:
        A DocumentIndex with entries for each contiguous section.
    """
    if not page_types:
        return DocumentIndex(source="generated")

    entries: list[TOCEntry] = []
    current_type = page_types[0][1]
    current_start = page_types[0][0]

    for page_num, doc_type in page_types[1:]:
        if doc_type != current_type:
            entries.append(TOCEntry(
                label=current_type.value,
                start_page=current_start,
                end_page=page_num - 1,
                mapped_type=current_type,
            ))
            current_type = doc_type
            current_start = page_num

    # Final section
    entries.append(TOCEntry(
        label=current_type.value,
        start_page=current_start,
        end_page=page_types[-1][0],
        mapped_type=current_type,
    ))

    logger.info("Generated index: %d sections from %d pages", len(entries), len(page_types))
    return DocumentIndex(entries=entries, source="generated")
