"""Source type classification: Primary vs Secondary source determination.

Applies the rules for distinguishing primary sources (original, first-hand
evidence) from secondary sources (summaries, interpretations, analyses).
"""

from __future__ import annotations

import re

from pdf_classifier.extractor import PageContent
from pdf_classifier.taxonomy import (
    ALWAYS_SECONDARY_TYPES,
    DEFAULT_PRIMARY_TYPES,
    DocumentType,
    SourceType,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _references_other_practitioner(text: str) -> bool:
    """Detect if the document describes another practitioner's consultation.

    A document referencing another practitioner's findings/consultations
    (rather than the author's own) is a secondary source.
    """
    patterns = [
        r"as reported by dr",
        r"as noted by dr",
        r"according to dr",
        r"dr [\w]+ reported",
        r"dr [\w]+ noted",
        r"dr [\w]+ found",
        r"seen by dr [\w]+ on",
        r"reviewed by dr [\w]+ on",
        r"consultation with dr [\w]+",
        r"records from dr",
        r"notes from dr",
        r"report from dr",
        r"previous assessment by",
        r"was seen at .+ hospital",
        r"was admitted to .+ hospital",
        r"attended .+ hospital",
        r"presented to .+ emergency",
        r"records indicate",
        r"notes indicate",
        r"records show",
        r"notes show",
        r"history obtained from",
        r"collateral information",
    ]
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False


def _summarizes_multiple_consultations(text: str) -> bool:
    """Detect if a treater letter summarizes multiple consultations.

    A treater's letter summarizing multiple consultations or treatment over
    a period is a secondary source, while one summarizing a single
    presentation by that same practitioner is primary.
    """
    patterns = [
        r"over the (?:past|last) \d+ (?:months?|years?|weeks?)",
        r"has been (?:attending|seeing|treating|managing)",
        r"multiple (?:consultations|appointments|visits|sessions)",
        r"series of (?:consultations|appointments|treatments)",
        r"treatment (?:history|summary|overview)",
        r"history of treatment",
        r"course of treatment",
        r"summary of (?:treatment|care|consultations)",
        r"has been under my care since",
        r"i have been (?:treating|seeing|managing)",
        r"(?:first|initially) (?:seen|presented|attended) (?:on|in)",
        r"treatment to date",
        r"progress to date",
        r"treatment from .+ to",
        r"seen on \d+.*seen (?:again |)on \d+",
    ]
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False


def _describes_hospital_presentation(text: str) -> bool:
    """Detect if a non-hospital document describes a hospital presentation.

    Non-hospital documents describing hospital presentations are secondary.
    """
    patterns = [
        r"attended .+ (?:hospital|emergency)",
        r"presented to .+ (?:hospital|emergency)",
        r"admitted to .+ hospital",
        r"was seen at .+ (?:hospital|emergency)",
        r"hospital records",
        r"hospital notes",
        r"discharge (?:summary|report) (?:from|indicates|states|shows)",
        r"hospital presentation on",
    ]
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False


def classify_source_type(
    document_type: DocumentType, page: PageContent
) -> tuple[SourceType, str]:
    """Determine whether a page is a primary or secondary source.

    Args:
        document_type: The classified document type for this page.
        page: The extracted page content.

    Returns:
        Tuple of (SourceType, reasoning_string).
    """
    # Always-secondary types need no further analysis
    if document_type in ALWAYS_SECONDARY_TYPES:
        return (
            SourceType.SECONDARY,
            f"{document_type.value} is always classified as secondary source",
        )

    text = _normalize(page.text)
    reasons: list[str] = []

    # Check context-dependent secondary indicators
    is_secondary = False

    # Check if document references another practitioner's work
    if _references_other_practitioner(text):
        is_secondary = True
        reasons.append(
            "references another practitioner's consultation/findings"
        )

    # For treater letters: check if summarizing multiple consultations
    if document_type in (
        DocumentType.TREATER_LETTER_NON_ALLIED,
        DocumentType.TREATER_LETTER_ALLIED,
    ):
        if _summarizes_multiple_consultations(text):
            is_secondary = True
            reasons.append(
                "treater letter summarizes multiple consultations/treatment period"
            )

    # For non-hospital documents: check if describing hospital presentations
    if document_type not in (
        DocumentType.ED_DISCHARGE_SUMMARY,
        DocumentType.ED_OPERATION_REPORT,
        DocumentType.ED_RADIOLOGY,
        DocumentType.ED_OUTPATIENT_NOTES,
        DocumentType.ED_OTHER,
    ):
        if _describes_hospital_presentation(text):
            is_secondary = True
            reasons.append(
                "non-hospital document describes hospital presentation"
            )

    if is_secondary:
        return (SourceType.SECONDARY, "; ".join(reasons))

    # Default: use the type-based default
    if document_type in DEFAULT_PRIMARY_TYPES:
        return (
            SourceType.PRIMARY,
            f"{document_type.value} defaults to primary source",
        )

    return (
        SourceType.PRIMARY,
        "No secondary source indicators detected; defaulting to primary",
    )
