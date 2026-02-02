"""Tests for the downstream processing router."""

import pytest

from pdf_classifier.models import PageClassification
from pdf_classifier.processor import (
    ProcessingResult,
    get_pages_by_workflow,
    route_batch,
    route_page,
)
from pdf_classifier.taxonomy import (
    DocumentType,
    PriorityLevel,
    SourceType,
)


def _make_classified_page(
    page_number: int = 1,
    doc_type: DocumentType = DocumentType.ED_DISCHARGE_SUMMARY,
    source: SourceType = SourceType.PRIMARY,
    priority: PriorityLevel = PriorityLevel.HIGH,
    text: str = "Sample extracted text with diagnosis and medications",
) -> PageClassification:
    return PageClassification(
        page_number=page_number,
        document_type=doc_type,
        source_type=source,
        priority_level=priority,
        confidence=0.8,
        extracted_text=text,
        classification_reasoning="Test reasoning",
    )


class TestRoutePageMapping:
    """Verify each document type routes to the correct processor."""

    def test_ed_discharge_routing(self):
        page = _make_classified_page(doc_type=DocumentType.ED_DISCHARGE_SUMMARY)
        result = route_page(page)
        assert result.success
        assert result.processor_name == "ed_discharge_processor"

    def test_ed_operation_routing(self):
        page = _make_classified_page(doc_type=DocumentType.ED_OPERATION_REPORT)
        result = route_page(page)
        assert result.success
        assert result.processor_name == "ed_operation_processor"

    def test_ed_radiology_routing(self):
        page = _make_classified_page(doc_type=DocumentType.ED_RADIOLOGY)
        result = route_page(page)
        assert result.success
        assert result.processor_name == "ed_radiology_processor"

    def test_ed_outpatient_routing(self):
        page = _make_classified_page(doc_type=DocumentType.ED_OUTPATIENT_NOTES)
        result = route_page(page)
        assert result.success
        assert result.processor_name == "ed_outpatient_processor"

    def test_ed_other_routing(self):
        page = _make_classified_page(doc_type=DocumentType.ED_OTHER)
        result = route_page(page)
        assert result.success
        assert result.processor_name == "ed_other_processor"

    def test_standalone_radiology_routing(self):
        page = _make_classified_page(doc_type=DocumentType.RADIOLOGY)
        result = route_page(page)
        assert result.success
        assert result.processor_name == "standalone_radiology_processor"

    def test_standalone_operation_routing(self):
        page = _make_classified_page(doc_type=DocumentType.OPERATION_REPORT)
        result = route_page(page)
        assert result.success
        assert result.processor_name == "standalone_operation_processor"

    def test_treater_letter_non_allied_routing(self):
        page = _make_classified_page(
            doc_type=DocumentType.TREATER_LETTER_NON_ALLIED
        )
        result = route_page(page)
        assert result.success
        assert result.processor_name == "treater_letter_processor"

    def test_treater_letter_allied_routing(self):
        page = _make_classified_page(
            doc_type=DocumentType.TREATER_LETTER_ALLIED
        )
        result = route_page(page)
        assert result.success
        assert result.processor_name == "treater_letter_processor"

    def test_referral_letter_routing(self):
        page = _make_classified_page(doc_type=DocumentType.REFERRAL_LETTER)
        result = route_page(page)
        assert result.success
        assert result.processor_name == "referral_letter_processor"

    def test_clinical_notes_non_allied_routing(self):
        page = _make_classified_page(
            doc_type=DocumentType.CLINICAL_NOTES_NON_ALLIED
        )
        result = route_page(page)
        assert result.success
        assert result.processor_name == "clinical_notes_processor"

    def test_clinical_notes_allied_routing(self):
        page = _make_classified_page(
            doc_type=DocumentType.CLINICAL_NOTES_ALLIED
        )
        result = route_page(page)
        assert result.success
        assert result.processor_name == "clinical_notes_processor"

    def test_specialist_clinical_notes_routing(self):
        page = _make_classified_page(
            doc_type=DocumentType.SPECIALIST_CLINICAL_NOTES
        )
        result = route_page(page)
        assert result.success
        assert result.processor_name == "clinical_notes_processor"

    def test_medicolegal_report_routing(self):
        page = _make_classified_page(doc_type=DocumentType.MEDICOLEGAL_REPORT)
        result = route_page(page)
        assert result.success
        assert result.processor_name == "medicolegal_report_processor"

    def test_medical_panel_routing(self):
        page = _make_classified_page(doc_type=DocumentType.MEDICAL_PANEL_REPORT)
        result = route_page(page)
        assert result.success
        assert result.processor_name == "medical_panel_processor"

    def test_affidavit_routing(self):
        page = _make_classified_page(doc_type=DocumentType.AFFIDAVIT)
        result = route_page(page)
        assert result.success
        assert result.processor_name == "affidavit_processor"

    def test_other_routing(self):
        page = _make_classified_page(doc_type=DocumentType.OTHER)
        result = route_page(page)
        assert result.success
        assert result.processor_name == "fallback_processor"


class TestProcessingResultData:
    def test_result_includes_classification_metadata(self):
        page = _make_classified_page(
            doc_type=DocumentType.ED_DISCHARGE_SUMMARY,
            source=SourceType.PRIMARY,
            priority=PriorityLevel.HIGH,
        )
        result = route_page(page)
        assert result.output_data["source_type"] == "Primary"
        assert result.output_data["priority_level"] == "High"
        assert result.output_data["category"] == "Hospital/ED Presentations"

    def test_discharge_processor_extracts_fields(self):
        page = _make_classified_page(
            doc_type=DocumentType.ED_DISCHARGE_SUMMARY,
            text="Discharge summary with diagnosis and follow up plan",
        )
        result = route_page(page)
        assert "extract_fields" in result.output_data
        assert result.output_data["action"] == "extract_discharge_data"
        assert result.output_data["has_diagnosis"]
        assert result.output_data["has_medications"] is False
        assert result.output_data["has_follow_up"]

    def test_radiology_processor_detects_emg(self):
        page = _make_classified_page(
            doc_type=DocumentType.RADIOLOGY,
            text="EMG and nerve conduction study findings",
        )
        result = route_page(page)
        assert result.output_data["is_emg_ncv"]


class TestBatchRouting:
    def test_route_batch_processes_all_pages(self):
        pages = [
            _make_classified_page(1, DocumentType.ED_DISCHARGE_SUMMARY),
            _make_classified_page(2, DocumentType.RADIOLOGY),
            _make_classified_page(3, DocumentType.MEDICOLEGAL_REPORT),
        ]
        results = route_batch(pages)
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_route_batch_preserves_page_order(self):
        pages = [
            _make_classified_page(1),
            _make_classified_page(2),
            _make_classified_page(3),
        ]
        results = route_batch(pages)
        for i, r in enumerate(results):
            assert r.page_number == i + 1


class TestWorkflowGrouping:
    def test_pages_grouped_by_workflow(self):
        pages = [
            _make_classified_page(1, DocumentType.ED_DISCHARGE_SUMMARY),
            _make_classified_page(2, DocumentType.ED_DISCHARGE_SUMMARY),
            _make_classified_page(3, DocumentType.RADIOLOGY),
            _make_classified_page(4, DocumentType.MEDICOLEGAL_REPORT),
        ]
        workflows = get_pages_by_workflow(pages)
        assert len(workflows["process_ed_discharge"]) == 2
        assert len(workflows["process_standalone_radiology"]) == 1
        assert len(workflows["process_medicolegal_report"]) == 1
