"""Tests for priority/importance level assignment."""

import pytest

from pdf_classifier.priority import assign_priority
from pdf_classifier.taxonomy import DocumentType, PriorityLevel, SourceType


class TestPrimarySourcePriorities:
    def test_ed_discharge_high(self):
        p = assign_priority(DocumentType.ED_DISCHARGE_SUMMARY, SourceType.PRIMARY)
        assert p == PriorityLevel.HIGH

    def test_radiology_high(self):
        p = assign_priority(DocumentType.RADIOLOGY, SourceType.PRIMARY)
        assert p == PriorityLevel.HIGH

    def test_operation_report_high(self):
        p = assign_priority(DocumentType.OPERATION_REPORT, SourceType.PRIMARY)
        assert p == PriorityLevel.HIGH

    def test_treater_letter_non_allied_high(self):
        p = assign_priority(
            DocumentType.TREATER_LETTER_NON_ALLIED, SourceType.PRIMARY
        )
        assert p == PriorityLevel.HIGH

    def test_ed_outpatient_mid(self):
        p = assign_priority(DocumentType.ED_OUTPATIENT_NOTES, SourceType.PRIMARY)
        assert p == PriorityLevel.MID

    def test_clinical_notes_non_allied_mid(self):
        p = assign_priority(
            DocumentType.CLINICAL_NOTES_NON_ALLIED, SourceType.PRIMARY
        )
        assert p == PriorityLevel.MID

    def test_allied_health_letter_mid(self):
        p = assign_priority(
            DocumentType.TREATER_LETTER_ALLIED, SourceType.PRIMARY
        )
        assert p == PriorityLevel.MID

    def test_referral_letter_mid(self):
        p = assign_priority(DocumentType.REFERRAL_LETTER, SourceType.PRIMARY)
        assert p == PriorityLevel.MID


class TestSecondarySourcePriorities:
    def test_medical_panel_high(self):
        p = assign_priority(
            DocumentType.MEDICAL_PANEL_REPORT, SourceType.SECONDARY
        )
        assert p == PriorityLevel.HIGH

    def test_affidavit_mid(self):
        p = assign_priority(DocumentType.AFFIDAVIT, SourceType.SECONDARY)
        assert p == PriorityLevel.MID

    def test_medicolegal_low(self):
        p = assign_priority(
            DocumentType.MEDICOLEGAL_REPORT, SourceType.SECONDARY
        )
        assert p == PriorityLevel.LOW


class TestDemotedSecondaryPriorities:
    """When a normally-primary type is reclassified as secondary,
    its priority should drop one level."""

    def test_high_primary_demoted_to_mid(self):
        # ED_DISCHARGE_SUMMARY is High as primary, should be Mid as secondary
        p = assign_priority(
            DocumentType.ED_DISCHARGE_SUMMARY, SourceType.SECONDARY
        )
        assert p == PriorityLevel.MID

    def test_mid_primary_demoted_to_low(self):
        # CLINICAL_NOTES_NON_ALLIED is Mid as primary, should be Low as secondary
        p = assign_priority(
            DocumentType.CLINICAL_NOTES_NON_ALLIED, SourceType.SECONDARY
        )
        assert p == PriorityLevel.LOW


class TestFallbackPriority:
    def test_other_type_primary_is_low(self):
        p = assign_priority(DocumentType.OTHER, SourceType.PRIMARY)
        assert p == PriorityLevel.LOW

    def test_other_type_secondary_is_low(self):
        p = assign_priority(DocumentType.OTHER, SourceType.SECONDARY)
        assert p == PriorityLevel.LOW
