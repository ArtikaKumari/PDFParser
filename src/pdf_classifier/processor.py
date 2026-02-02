"""Downstream processing router and per-type processing workflows.

Routes classified pages to specific processing functions based on their
document type, source type, and priority level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from pdf_classifier.models import PageClassification
from pdf_classifier.taxonomy import (
    DocumentCategory,
    DocumentType,
    PriorityLevel,
    SourceType,
    DOCUMENT_TYPE_TO_CATEGORY,
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result from a downstream processing function."""

    page_number: int
    processor_name: str
    success: bool
    output_data: dict = field(default_factory=dict)
    error_message: str = ""


# Type alias for processing functions
ProcessingFn = Callable[[PageClassification], ProcessingResult]


def _make_result(
    page: PageClassification,
    name: str,
    data: dict | None = None,
) -> ProcessingResult:
    return ProcessingResult(
        page_number=page.page_number,
        processor_name=name,
        success=True,
        output_data=data or {},
    )


# ---------------------------------------------------------------------------
# Processing functions for each document category / type
# ---------------------------------------------------------------------------


def process_ed_discharge(page: PageClassification) -> ProcessingResult:
    """Process ED/Hospital discharge summaries.

    Extracts structured data: diagnoses, procedures, medications,
    follow-up instructions.
    """
    text = page.extracted_text.lower()
    data: dict = {
        "action": "extract_discharge_data",
        "extract_fields": [
            "principal_diagnosis",
            "secondary_diagnoses",
            "procedures",
            "discharge_medications",
            "follow_up_plan",
            "length_of_stay",
        ],
        "has_diagnosis": "diagnosis" in text or "diagnos" in text,
        "has_medications": "medication" in text or "prescription" in text,
        "has_follow_up": "follow up" in text or "follow-up" in text,
    }
    return _make_result(page, "ed_discharge_processor", data)


def process_ed_operation(page: PageClassification) -> ProcessingResult:
    """Process ED/Hospital operation reports.

    Extracts surgical details: procedure type, findings, surgeon info.
    """
    text = page.extracted_text.lower()
    data: dict = {
        "action": "extract_operative_data",
        "extract_fields": [
            "procedure_name",
            "surgeon",
            "anaesthetist",
            "operative_findings",
            "post_operative_instructions",
        ],
        "has_findings": "findings" in text,
        "has_surgeon": "surgeon" in text,
    }
    return _make_result(page, "ed_operation_processor", data)


def process_ed_radiology(page: PageClassification) -> ProcessingResult:
    """Process ED/Hospital radiology reports."""
    text = page.extracted_text.lower()
    data: dict = {
        "action": "extract_radiology_data",
        "extract_fields": [
            "modality",
            "body_region",
            "clinical_indication",
            "findings",
            "impression",
            "radiologist",
        ],
        "has_impression": "impression" in text,
        "has_findings": "findings" in text,
    }
    return _make_result(page, "ed_radiology_processor", data)


def process_ed_outpatient(page: PageClassification) -> ProcessingResult:
    """Process ED/Hospital outpatient consultation notes."""
    data: dict = {
        "action": "extract_outpatient_data",
        "extract_fields": [
            "clinic_type",
            "clinician",
            "assessment",
            "plan",
            "next_review_date",
        ],
    }
    return _make_result(page, "ed_outpatient_processor", data)


def process_ed_other(page: PageClassification) -> ProcessingResult:
    """Process other ED/Hospital documents (progress notes, charts, etc.)."""
    data: dict = {
        "action": "archive_hospital_document",
        "extract_fields": ["document_subtype", "date", "author"],
    }
    return _make_result(page, "ed_other_processor", data)


def process_standalone_radiology(page: PageClassification) -> ProcessingResult:
    """Process standalone radiology reports (including EMG/NCV)."""
    text = page.extracted_text.lower()
    data: dict = {
        "action": "extract_radiology_data",
        "extract_fields": [
            "modality",
            "body_region",
            "clinical_indication",
            "findings",
            "impression",
            "radiologist",
            "referring_clinician",
        ],
        "is_emg_ncv": "emg" in text or "nerve conduction" in text or "ncv" in text,
        "has_impression": "impression" in text,
    }
    return _make_result(page, "standalone_radiology_processor", data)


def process_standalone_operation(page: PageClassification) -> ProcessingResult:
    """Process standalone operation reports."""
    data: dict = {
        "action": "extract_operative_data",
        "extract_fields": [
            "procedure_name",
            "surgeon",
            "facility",
            "operative_findings",
            "complications",
        ],
    }
    return _make_result(page, "standalone_operation_processor", data)


def process_treater_letter(page: PageClassification) -> ProcessingResult:
    """Process treater letters (allied and non-allied health)."""
    data: dict = {
        "action": "extract_correspondence_data",
        "extract_fields": [
            "author",
            "author_specialty",
            "addressee",
            "date",
            "patient_name",
            "summary_of_findings",
            "recommendations",
        ],
        "letter_subtype": page.document_type.value,
    }
    return _make_result(page, "treater_letter_processor", data)


def process_referral_letter(page: PageClassification) -> ProcessingResult:
    """Process referral letters."""
    data: dict = {
        "action": "extract_referral_data",
        "extract_fields": [
            "referring_practitioner",
            "referred_to",
            "reason_for_referral",
            "clinical_summary",
            "urgency",
        ],
    }
    return _make_result(page, "referral_letter_processor", data)


def process_clinical_notes(page: PageClassification) -> ProcessingResult:
    """Process clinical notes (allied, non-allied, specialist)."""
    data: dict = {
        "action": "extract_clinical_notes_data",
        "extract_fields": [
            "clinician",
            "date_of_consultation",
            "presenting_complaint",
            "examination_findings",
            "assessment",
            "plan",
        ],
        "notes_subtype": page.document_type.value,
    }
    return _make_result(page, "clinical_notes_processor", data)


def process_medicolegal_report(page: PageClassification) -> ProcessingResult:
    """Process medicolegal reports.

    Extracts key medicolegal opinions, impairment assessments, and
    causation findings.
    """
    data: dict = {
        "action": "extract_medicolegal_data",
        "extract_fields": [
            "examiner",
            "requesting_party",
            "date_of_examination",
            "diagnoses",
            "causation_opinion",
            "impairment_rating",
            "work_capacity_opinion",
            "treatment_recommendations",
            "prognosis",
        ],
    }
    return _make_result(page, "medicolegal_report_processor", data)


def process_medical_panel_report(page: PageClassification) -> ProcessingResult:
    """Process medical panel reports."""
    data: dict = {
        "action": "extract_panel_report_data",
        "extract_fields": [
            "panel_members",
            "date_of_examination",
            "questions_posed",
            "panel_opinions",
            "impairment_assessment",
        ],
    }
    return _make_result(page, "medical_panel_processor", data)


def process_affidavit(page: PageClassification) -> ProcessingResult:
    """Process affidavits."""
    data: dict = {
        "action": "extract_affidavit_data",
        "extract_fields": [
            "deponent",
            "date_sworn",
            "court_reference",
            "key_statements",
        ],
    }
    return _make_result(page, "affidavit_processor", data)


def process_other(page: PageClassification) -> ProcessingResult:
    """Fallback processor for unclassified documents."""
    data: dict = {
        "action": "flag_for_manual_review",
        "extract_fields": ["date", "author", "content_summary"],
    }
    return _make_result(page, "fallback_processor", data)


# ---------------------------------------------------------------------------
# Routing table: maps DocumentType -> processing function
# ---------------------------------------------------------------------------

_PROCESSING_ROUTES: dict[DocumentType, ProcessingFn] = {
    DocumentType.ED_DISCHARGE_SUMMARY: process_ed_discharge,
    DocumentType.ED_OPERATION_REPORT: process_ed_operation,
    DocumentType.ED_RADIOLOGY: process_ed_radiology,
    DocumentType.ED_OUTPATIENT_NOTES: process_ed_outpatient,
    DocumentType.ED_OTHER: process_ed_other,
    DocumentType.RADIOLOGY: process_standalone_radiology,
    DocumentType.OPERATION_REPORT: process_standalone_operation,
    DocumentType.TREATER_LETTER_NON_ALLIED: process_treater_letter,
    DocumentType.TREATER_LETTER_ALLIED: process_treater_letter,
    DocumentType.REFERRAL_LETTER: process_referral_letter,
    DocumentType.CLINICAL_NOTES_NON_ALLIED: process_clinical_notes,
    DocumentType.CLINICAL_NOTES_ALLIED: process_clinical_notes,
    DocumentType.SPECIALIST_CLINICAL_NOTES: process_clinical_notes,
    DocumentType.MEDICOLEGAL_REPORT: process_medicolegal_report,
    DocumentType.MEDICAL_PANEL_REPORT: process_medical_panel_report,
    DocumentType.AFFIDAVIT: process_affidavit,
    DocumentType.OTHER: process_other,
}


def route_page(page: PageClassification) -> ProcessingResult:
    """Route a classified page to its appropriate processing function.

    Args:
        page: A fully classified page (with document type, source type,
              and priority level already assigned).

    Returns:
        ProcessingResult from the appropriate downstream processor.
    """
    processor = _PROCESSING_ROUTES.get(page.document_type, process_other)
    try:
        result = processor(page)
        # Enrich with classification metadata
        result.output_data["source_type"] = page.source_type.value
        result.output_data["priority_level"] = page.priority_level.value
        result.output_data["category"] = page.category.value
        logger.info(
            "Page %d processed by %s",
            page.page_number,
            result.processor_name,
        )
        return result
    except Exception as exc:
        logger.exception(
            "Processing failed for page %d", page.page_number
        )
        return ProcessingResult(
            page_number=page.page_number,
            processor_name=processor.__name__,
            success=False,
            error_message=str(exc),
        )


def route_batch(
    pages: list[PageClassification],
) -> list[ProcessingResult]:
    """Route a batch of classified pages to their processing functions.

    Args:
        pages: List of classified pages.

    Returns:
        List of ProcessingResult objects, one per page.
    """
    results: list[ProcessingResult] = []
    for page in pages:
        result = route_page(page)
        results.append(result)
    return results


def get_pages_by_workflow(
    pages: list[PageClassification],
) -> dict[str, list[PageClassification]]:
    """Group pages by their processing workflow for batch processing.

    Returns a dictionary mapping processor names to the pages that
    should be processed by that processor.
    """
    workflows: dict[str, list[PageClassification]] = {}
    for page in pages:
        processor = _PROCESSING_ROUTES.get(page.document_type, process_other)
        name = processor.__name__
        if name not in workflows:
            workflows[name] = []
        workflows[name].append(page)
    return workflows
