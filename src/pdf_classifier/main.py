"""Main application orchestrator and CLI entry point.

Coordinates PDF extraction, classification, source typing, priority
assignment, downstream processing, and output generation.

Supports a TOC-first classification flow: if a table of contents / index
page is detected in the PDF, the TOC entries are used to pre-classify
pages.  Pages not covered by the TOC (or all pages when no TOC exists)
fall back to the keyword-based classifier.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pdf_classifier.classifier import classify_page
from pdf_classifier.extractor import PageContent, extract_all_pages
from pdf_classifier.models import DocumentResult, PageClassification
from pdf_classifier.output import export_csv, export_json, format_report
from pdf_classifier.priority import assign_priority
from pdf_classifier.processor import ProcessingResult, route_batch
from pdf_classifier.source_classifier import classify_source_type
from pdf_classifier.taxonomy import DocumentType, SourceType, PriorityLevel
from pdf_classifier.toc_parser import (
    DocumentIndex,
    build_index_from_toc,
    generate_index_from_classifications,
)

logger = logging.getLogger(__name__)

# Pages with confidence below this threshold get flagged for manual review
_REVIEW_THRESHOLD = 0.35

# Confidence score assigned to pages classified via TOC lookup
_TOC_GUIDED_CONFIDENCE = 0.85


def classify_document(pdf_path: str | Path) -> DocumentResult:
    """Run the full classification pipeline on a PDF document.

    The pipeline uses a TOC-first strategy:
      1. Extract text from all pages.
      2. Search for a Table of Contents / index page.
      3a. If a TOC is found, use it to pre-classify pages that are covered
          by TOC entries.  TOC pages themselves are classified as TOC type.
      3b. Pages not covered by the TOC (or all pages when no TOC exists)
          are classified using the keyword-based scoring engine.
      4. Source type and priority are assigned for every page.
      5. A synthetic document index is generated when no TOC was found.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        DocumentResult with per-page classifications.
    """
    path = Path(pdf_path)
    logger.info("Starting classification of: %s", path)

    # Step 1: Extract text from all pages
    pages: list[PageContent] = extract_all_pages(path)
    result = DocumentResult(file_path=str(path), total_pages=len(pages))

    # Step 2: Attempt to find and parse a TOC
    doc_index = build_index_from_toc(pages, len(pages))

    if doc_index:
        logger.info(
            "TOC found (%d entries). Using TOC-guided classification.",
            len(doc_index.entries),
        )

    # Step 3-5: Classify each page
    for page in pages:
        try:
            classification = _classify_single_page(page, doc_index)
            result.page_classifications.append(classification)
        except Exception as exc:
            error_msg = f"Page {page.page_number}: classification failed - {exc}"
            logger.exception(error_msg)
            result.processing_errors.append(error_msg)
            # Add a fallback classification for the failed page
            result.page_classifications.append(
                PageClassification(
                    page_number=page.page_number,
                    document_type=DocumentType.OTHER,
                    source_type=classify_source_type(DocumentType.OTHER, page)[0],
                    priority_level=assign_priority(
                        DocumentType.OTHER,
                        classify_source_type(DocumentType.OTHER, page)[0],
                    ),
                    confidence=0.0,
                    extracted_text=page.text,
                    classification_reasoning=f"Classification error: {exc}",
                    flags=["classification_error"],
                    needs_manual_review=True,
                )
            )

    # Step 6: If no TOC was found, generate a synthetic index from results
    if not doc_index:
        page_types = [
            (pc.page_number, pc.document_type)
            for pc in result.page_classifications
        ]
        doc_index = generate_index_from_classifications(page_types)
        logger.info(
            "No TOC found. Generated synthetic index with %d sections.",
            len(doc_index.entries),
        )

    logger.info(
        "Classification complete: %d pages processed, %d errors",
        len(result.page_classifications),
        len(result.processing_errors),
    )
    return result


def _classify_single_page(
    page: PageContent,
    doc_index: DocumentIndex | None = None,
) -> PageClassification:
    """Run the full classification pipeline on a single page.

    If *doc_index* is provided (i.e. a TOC was found):
      - TOC pages themselves are classified as ``DocumentType.TOC``.
      - Pages covered by TOC entries use the mapped type from the TOC.
      - Pages not covered by the TOC fall through to keyword classification.

    Args:
        page: Extracted page content.
        doc_index: Optional document index built from a TOC.

    Returns:
        A fully classified PageClassification.
    """
    doc_type: DocumentType | None = None
    confidence: float = 0.0
    type_reasoning: str = ""

    # --- TOC-guided classification ---
    if doc_index and doc_index.source == "toc":
        # Mark TOC pages
        if doc_index.is_toc_page(page.page_number):
            doc_type = DocumentType.TOC
            confidence = 1.0
            type_reasoning = "Page is a Table of Contents / index page"
        else:
            # Look up from TOC entries
            toc_type = doc_index.get_type_for_page(page.page_number)
            if toc_type is not None:
                doc_type = toc_type
                confidence = _TOC_GUIDED_CONFIDENCE
                type_reasoning = (
                    f"Classified via TOC lookup as {toc_type.value}"
                )

    # --- Fallback to keyword classifier ---
    if doc_type is None:
        doc_type, confidence, type_reasoning = classify_page(page)

    # Source type
    source_type, source_reasoning = classify_source_type(doc_type, page)

    # Priority
    priority = assign_priority(doc_type, source_type)

    # Flags
    flags: list[str] = []
    needs_review = False

    if confidence < _REVIEW_THRESHOLD:
        flags.append("low_confidence")
        needs_review = True

    if page.extraction_method == "ocr":
        flags.append("ocr_extracted")

    if page.extraction_method == "empty":
        flags.append("no_text_content")
        needs_review = True

    if page.word_count < 10:
        flags.append("very_short_content")

    if doc_index and doc_index.source == "toc" and doc_type != DocumentType.TOC:
        toc_type = doc_index.get_type_for_page(page.page_number)
        if toc_type is not None:
            flags.append("toc_guided")

    reasoning = f"Type: {type_reasoning} | Source: {source_reasoning}"

    return PageClassification(
        page_number=page.page_number,
        document_type=doc_type,
        source_type=source_type,
        priority_level=priority,
        confidence=confidence,
        extracted_text=page.text,
        classification_reasoning=reasoning,
        flags=flags,
        needs_manual_review=needs_review,
    )


def process_document(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    output_format: str = "text",
) -> tuple[DocumentResult, list[ProcessingResult]]:
    """Full pipeline: classify and then process a PDF document.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Optional directory for output files.
        output_format: Output format - "text", "json", "csv", or "all".

    Returns:
        Tuple of (DocumentResult, list of ProcessingResults).
    """
    # Classify
    result = classify_document(pdf_path)

    # Process through downstream handlers
    processing_results = route_batch(result.page_classifications)

    # Generate output
    report = format_report(result, processing_results)

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        stem = Path(pdf_path).stem

        if output_format in ("text", "all"):
            text_path = out_path / f"{stem}_report.txt"
            text_path.write_text(report, encoding="utf-8")
            logger.info("Text report saved to %s", text_path)

        if output_format in ("json", "all"):
            json_path = out_path / f"{stem}_results.json"
            export_json(result, json_path)

        if output_format in ("csv", "all"):
            csv_path = out_path / f"{stem}_results.csv"
            export_csv(result, csv_path)
    else:
        print(report)

    return result, processing_results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-classifier",
        description=(
            "Classify pages of medicolegal enclosure PDF documents "
            "and route them to appropriate processing workflows."
        ),
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file to process",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Directory for output files (default: print to stdout)",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["text", "json", "csv", "all"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    parser.add_argument(
        "--classify-only",
        action="store_true",
        help="Only classify pages, skip downstream processing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        return 1

    if pdf_path.suffix.lower() != ".pdf":
        print(f"Error: not a PDF file: {pdf_path}", file=sys.stderr)
        return 1

    try:
        if args.classify_only:
            result = classify_document(pdf_path)
            report = format_report(result)
            if args.output_dir:
                out_path = Path(args.output_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                stem = pdf_path.stem
                if args.format in ("json", "all"):
                    export_json(result, out_path / f"{stem}_results.json")
                if args.format in ("csv", "all"):
                    export_csv(result, out_path / f"{stem}_results.csv")
                if args.format in ("text", "all"):
                    text_path = out_path / f"{stem}_report.txt"
                    text_path.write_text(report, encoding="utf-8")
                    logger.info("Text report saved to %s", text_path)
            else:
                print(report)
        else:
            process_document(pdf_path, args.output_dir, args.format)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error processing %s", pdf_path)
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
