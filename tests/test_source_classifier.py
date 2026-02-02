"""Tests for primary/secondary source type classification."""

import pytest

from pdf_classifier.extractor import PageContent
from pdf_classifier.source_classifier import classify_source_type
from pdf_classifier.taxonomy import DocumentType, SourceType


def _make_page(text: str, page_number: int = 1) -> PageContent:
    return PageContent(
        page_number=page_number,
        text=text,
        extraction_method="text",
        has_images=False,
        word_count=len(text.split()),
    )


class TestAlwaysSecondaryTypes:
    def test_medicolegal_is_always_secondary(self):
        page = _make_page("Some medicolegal report text")
        source_type, _ = classify_source_type(
            DocumentType.MEDICOLEGAL_REPORT, page
        )
        assert source_type == SourceType.SECONDARY

    def test_medical_panel_is_always_secondary(self):
        page = _make_page("Medical panel report text")
        source_type, _ = classify_source_type(
            DocumentType.MEDICAL_PANEL_REPORT, page
        )
        assert source_type == SourceType.SECONDARY

    def test_affidavit_is_always_secondary(self):
        page = _make_page("Affidavit content")
        source_type, _ = classify_source_type(DocumentType.AFFIDAVIT, page)
        assert source_type == SourceType.SECONDARY


class TestDefaultPrimaryTypes:
    def test_discharge_summary_default_primary(self):
        page = _make_page("Discharge summary from hospital admission")
        source_type, _ = classify_source_type(
            DocumentType.ED_DISCHARGE_SUMMARY, page
        )
        assert source_type == SourceType.PRIMARY

    def test_radiology_default_primary(self):
        page = _make_page("MRI findings and impression")
        source_type, _ = classify_source_type(DocumentType.RADIOLOGY, page)
        assert source_type == SourceType.PRIMARY

    def test_clinical_notes_default_primary(self):
        page = _make_page("Clinical examination and assessment notes")
        source_type, _ = classify_source_type(
            DocumentType.CLINICAL_NOTES_NON_ALLIED, page
        )
        assert source_type == SourceType.PRIMARY


class TestContextDependentSecondary:
    def test_references_other_practitioner(self):
        """A document referencing another practitioner's work = secondary."""
        text = """
        Dr Smith reviewed the patient. According to Dr Jones,
        the patient had ongoing symptoms. Notes from Dr Williams
        indicate improvement.
        """
        page = _make_page(text)
        source_type, reasoning = classify_source_type(
            DocumentType.CLINICAL_NOTES_NON_ALLIED, page
        )
        assert source_type == SourceType.SECONDARY
        assert "another practitioner" in reasoning

    def test_treater_letter_single_visit_is_primary(self):
        """A treater's letter about a single visit = primary."""
        text = """
        Dear Dr Smith,
        I reviewed Mr Patient on 15/04/2024.
        On examination I found reduced range of motion.
        Yours sincerely, Dr Brown
        """
        page = _make_page(text)
        source_type, _ = classify_source_type(
            DocumentType.TREATER_LETTER_NON_ALLIED, page
        )
        assert source_type == SourceType.PRIMARY

    def test_treater_letter_multiple_consultations_is_secondary(self):
        """A treater's letter summarizing multiple visits = secondary."""
        text = """
        Dear Dr Smith,
        I have been treating Mr Patient over the past 6 months
        for his lower back condition. He has had multiple consultations
        and treatment sessions. Summary of treatment to date shows
        gradual improvement.
        Yours sincerely, Dr Brown
        """
        page = _make_page(text)
        source_type, reasoning = classify_source_type(
            DocumentType.TREATER_LETTER_NON_ALLIED, page
        )
        assert source_type == SourceType.SECONDARY
        assert "multiple consultations" in reasoning

    def test_non_hospital_describing_hospital_presentation(self):
        """A non-hospital document describing a hospital visit = secondary."""
        text = """
        Clinical notes from Dr Green.
        Patient attended Royal Melbourne Hospital emergency department
        on 01/03/2024. Hospital records indicate he was admitted for
        observation overnight.
        """
        page = _make_page(text)
        source_type, reasoning = classify_source_type(
            DocumentType.CLINICAL_NOTES_NON_ALLIED, page
        )
        assert source_type == SourceType.SECONDARY
        assert "hospital presentation" in reasoning

    def test_hospital_document_describing_own_presentation_is_primary(self):
        """Hospital documents describing their own presentations = primary."""
        text = """
        Emergency Department Discharge Summary
        Patient presented to this hospital with chest pain.
        Admitted for observation. Discharged next day.
        """
        page = _make_page(text)
        source_type, _ = classify_source_type(
            DocumentType.ED_DISCHARGE_SUMMARY, page
        )
        assert source_type == SourceType.PRIMARY


class TestSourceClassificationReasoning:
    def test_always_secondary_gives_clear_reasoning(self):
        page = _make_page("Report content")
        _, reasoning = classify_source_type(
            DocumentType.MEDICOLEGAL_REPORT, page
        )
        assert "always" in reasoning.lower()
        assert "secondary" in reasoning.lower()

    def test_primary_default_gives_clear_reasoning(self):
        page = _make_page("Simple clinical notes")
        _, reasoning = classify_source_type(
            DocumentType.CLINICAL_NOTES_NON_ALLIED, page
        )
        assert len(reasoning) > 0
