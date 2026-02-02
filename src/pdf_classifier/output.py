"""Output formatting and reporting for classification results."""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path

from pdf_classifier.models import DocumentResult, PageClassification
from pdf_classifier.processor import ProcessingResult
from pdf_classifier.taxonomy import DocumentCategory

logger = logging.getLogger(__name__)


def format_page_summary(page: PageClassification) -> str:
    """Format a single page classification as a human-readable summary line."""
    review_flag = " [NEEDS REVIEW]" if page.needs_manual_review else ""
    return (
        f"  Page {page.page_number:>3d}: "
        f"{page.document_type.value:<55s} "
        f"| {page.source_type.value:<10s} "
        f"| {page.priority_level.value:<5s} "
        f"| conf={page.confidence:.2f}"
        f"{review_flag}"
    )


def format_report(
    result: DocumentResult,
    processing_results: list[ProcessingResult] | None = None,
) -> str:
    """Format a complete document classification report.

    Args:
        result: The document classification result.
        processing_results: Optional downstream processing results.

    Returns:
        Formatted report string.
    """
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("PDF DOCUMENT CLASSIFICATION REPORT")
    lines.append("=" * 90)
    lines.append(f"File: {result.file_path}")
    lines.append(f"Total pages: {result.total_pages}")
    lines.append("")

    # Classification summary
    lines.append("--- Classification Summary ---")
    for doc_type, count in sorted(result.classification_summary.items()):
        lines.append(f"  {doc_type:<55s}: {count}")
    lines.append("")

    # Source type summary
    lines.append("--- Source Type Summary ---")
    for source_type, count in sorted(result.source_type_summary.items()):
        lines.append(f"  {source_type:<10s}: {count}")
    lines.append("")

    # Priority summary
    lines.append("--- Priority Summary ---")
    for priority, count in sorted(result.priority_summary.items()):
        lines.append(f"  {priority:<5s}: {count}")
    lines.append("")

    # Page-by-page details
    lines.append("--- Page-by-Page Classification ---")
    header = (
        f"  {'Page':>6s}: "
        f"{'Document Type':<55s} "
        f"| {'Source':<10s} "
        f"| {'Pri.':<5s} "
        f"| {'Conf.':<6s}"
    )
    lines.append(header)
    lines.append("  " + "-" * 86)
    for page in result.page_classifications:
        lines.append(format_page_summary(page))

    # Pages needing review
    review_pages = result.pages_needing_review
    if review_pages:
        lines.append("")
        lines.append("--- Pages Requiring Manual Review ---")
        for page in review_pages:
            lines.append(
                f"  Page {page.page_number}: {page.classification_reasoning}"
            )

    # Processing results (if provided)
    if processing_results:
        lines.append("")
        lines.append("--- Processing Results ---")
        for pr in processing_results:
            status = "OK" if pr.success else f"FAILED: {pr.error_message}"
            lines.append(
                f"  Page {pr.page_number:>3d}: "
                f"{pr.processor_name:<35s} [{status}]"
            )

    # Errors
    if result.processing_errors:
        lines.append("")
        lines.append("--- Processing Errors ---")
        for err in result.processing_errors:
            lines.append(f"  {err}")

    lines.append("")
    lines.append("=" * 90)
    return "\n".join(lines)


def export_csv(
    result: DocumentResult, output_path: str | Path
) -> None:
    """Export classification results to CSV.

    Args:
        result: The document classification result.
        output_path: Path for the CSV output file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "page_number",
        "document_type",
        "category",
        "source_type",
        "priority_level",
        "confidence",
        "needs_manual_review",
        "classification_reasoning",
        "flags",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for page in result.page_classifications:
            writer.writerow({
                "page_number": page.page_number,
                "document_type": page.document_type.value,
                "category": page.category.value,
                "source_type": page.source_type.value,
                "priority_level": page.priority_level.value,
                "confidence": round(page.confidence, 3),
                "needs_manual_review": page.needs_manual_review,
                "classification_reasoning": page.classification_reasoning,
                "flags": "; ".join(page.flags),
            })

    logger.info("CSV results exported to %s", path)


def export_json(
    result: DocumentResult, output_path: str | Path
) -> None:
    """Export classification results to JSON.

    Args:
        result: The document classification result.
        output_path: Path for the JSON output file.
    """
    result.save_json(output_path)


def export_grouped_by_category(
    result: DocumentResult,
) -> dict[str, list[dict]]:
    """Group classification results by document category.

    Returns a dictionary keyed by category name, with lists of page
    classification dicts under each.
    """
    grouped: dict[str, list[dict]] = {}
    for cat in DocumentCategory:
        pages = result.pages_by_category(cat)
        if pages:
            grouped[cat.value] = [p.to_dict() for p in pages]
    return grouped
