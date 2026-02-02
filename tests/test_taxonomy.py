"""Tests for taxonomy definitions and mappings."""

import pytest

from pdf_classifier.taxonomy import (
    ALWAYS_SECONDARY_TYPES,
    DEFAULT_PRIMARY_TYPES,
    DOCUMENT_TYPE_TO_CATEGORY,
    PRIMARY_SOURCE_PRIORITY,
    SECONDARY_SOURCE_PRIORITY,
    DocumentCategory,
    DocumentType,
    PriorityLevel,
    SourceType,
)


class TestDocumentTypeToCategory:
    """Verify every DocumentType has a category mapping."""

    def test_all_types_have_category(self):
        for doc_type in DocumentType:
            assert doc_type in DOCUMENT_TYPE_TO_CATEGORY, (
                f"{doc_type} missing from DOCUMENT_TYPE_TO_CATEGORY"
            )

    def test_hospital_ed_types(self):
        ed_types = [
            DocumentType.ED_DISCHARGE_SUMMARY,
            DocumentType.ED_OPERATION_REPORT,
            DocumentType.ED_RADIOLOGY,
            DocumentType.ED_OUTPATIENT_NOTES,
            DocumentType.ED_OTHER,
        ]
        for dt in ed_types:
            assert DOCUMENT_TYPE_TO_CATEGORY[dt] == DocumentCategory.HOSPITAL_ED

    def test_standalone_medical_types(self):
        assert (
            DOCUMENT_TYPE_TO_CATEGORY[DocumentType.RADIOLOGY]
            == DocumentCategory.STANDALONE_MEDICAL
        )
        assert (
            DOCUMENT_TYPE_TO_CATEGORY[DocumentType.OPERATION_REPORT]
            == DocumentCategory.STANDALONE_MEDICAL
        )

    def test_correspondence_types(self):
        corr_types = [
            DocumentType.TREATER_LETTER_NON_ALLIED,
            DocumentType.TREATER_LETTER_ALLIED,
            DocumentType.REFERRAL_LETTER,
        ]
        for dt in corr_types:
            assert DOCUMENT_TYPE_TO_CATEGORY[dt] == DocumentCategory.CORRESPONDENCE

    def test_clinical_documentation_types(self):
        clin_types = [
            DocumentType.CLINICAL_NOTES_NON_ALLIED,
            DocumentType.CLINICAL_NOTES_ALLIED,
            DocumentType.SPECIALIST_CLINICAL_NOTES,
        ]
        for dt in clin_types:
            assert (
                DOCUMENT_TYPE_TO_CATEGORY[dt]
                == DocumentCategory.CLINICAL_DOCUMENTATION
            )

    def test_legal_assessment_types(self):
        legal_types = [
            DocumentType.MEDICOLEGAL_REPORT,
            DocumentType.MEDICAL_PANEL_REPORT,
            DocumentType.AFFIDAVIT,
        ]
        for dt in legal_types:
            assert (
                DOCUMENT_TYPE_TO_CATEGORY[dt]
                == DocumentCategory.LEGAL_ASSESSMENT
            )

    def test_fallback_type(self):
        assert (
            DOCUMENT_TYPE_TO_CATEGORY[DocumentType.OTHER]
            == DocumentCategory.FALLBACK
        )


class TestSourceTypeDefaults:
    """Verify source type default sets are consistent."""

    def test_always_secondary_types(self):
        expected = {
            DocumentType.MEDICOLEGAL_REPORT,
            DocumentType.MEDICAL_PANEL_REPORT,
            DocumentType.AFFIDAVIT,
        }
        assert ALWAYS_SECONDARY_TYPES == expected

    def test_no_overlap_between_secondary_and_primary(self):
        overlap = ALWAYS_SECONDARY_TYPES & DEFAULT_PRIMARY_TYPES
        assert not overlap, f"Types in both sets: {overlap}"

    def test_all_non_other_types_categorized(self):
        all_classified = ALWAYS_SECONDARY_TYPES | DEFAULT_PRIMARY_TYPES
        for dt in DocumentType:
            if dt != DocumentType.OTHER:
                assert dt in all_classified, f"{dt} not in any source set"


class TestPriorityMappings:
    """Verify priority mappings are complete and valid."""

    def test_all_primary_default_types_have_priority(self):
        for dt in DEFAULT_PRIMARY_TYPES:
            assert dt in PRIMARY_SOURCE_PRIORITY, (
                f"{dt} missing from PRIMARY_SOURCE_PRIORITY"
            )

    def test_all_always_secondary_types_have_priority(self):
        for dt in ALWAYS_SECONDARY_TYPES:
            assert dt in SECONDARY_SOURCE_PRIORITY, (
                f"{dt} missing from SECONDARY_SOURCE_PRIORITY"
            )

    def test_medical_panel_is_high_priority(self):
        assert (
            SECONDARY_SOURCE_PRIORITY[DocumentType.MEDICAL_PANEL_REPORT]
            == PriorityLevel.HIGH
        )

    def test_affidavit_is_mid_priority(self):
        assert (
            SECONDARY_SOURCE_PRIORITY[DocumentType.AFFIDAVIT]
            == PriorityLevel.MID
        )

    def test_medicolegal_is_low_priority(self):
        assert (
            SECONDARY_SOURCE_PRIORITY[DocumentType.MEDICOLEGAL_REPORT]
            == PriorityLevel.LOW
        )
