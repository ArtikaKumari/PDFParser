"""Tests for the TOC parser, index builder, and TOC-first classification flow."""

import pytest

from pdf_classifier.extractor import PageContent
from pdf_classifier.taxonomy import DocumentType
from pdf_classifier.toc_parser import (
    TOCEntry,
    DocumentIndex,
    is_toc_page,
    find_toc_pages,
    parse_toc_entries,
    build_index_from_toc,
    generate_index_from_classifications,
    _map_label_to_type,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_page(text: str, page_number: int = 1) -> PageContent:
    return PageContent(
        page_number=page_number,
        text=text,
        extraction_method="text",
        has_images=False,
        word_count=len(text.split()),
    )


# ---------------------------------------------------------------------------
# TOCEntry & DocumentIndex dataclass tests
# ---------------------------------------------------------------------------


class TestDocumentIndex:
    def test_get_type_for_page_found(self):
        idx = DocumentIndex(
            entries=[
                TOCEntry("Discharge Summary", 1, 3, DocumentType.ED_DISCHARGE_SUMMARY),
                TOCEntry("Radiology", 4, 6, DocumentType.RADIOLOGY),
            ],
            source="toc",
        )
        assert idx.get_type_for_page(1) == DocumentType.ED_DISCHARGE_SUMMARY
        assert idx.get_type_for_page(3) == DocumentType.ED_DISCHARGE_SUMMARY
        assert idx.get_type_for_page(4) == DocumentType.RADIOLOGY
        assert idx.get_type_for_page(6) == DocumentType.RADIOLOGY

    def test_get_type_for_page_not_found(self):
        idx = DocumentIndex(
            entries=[
                TOCEntry("Discharge Summary", 2, 4, DocumentType.ED_DISCHARGE_SUMMARY),
            ],
            source="toc",
        )
        # Page 1 (before range) and page 5 (after range) not covered
        assert idx.get_type_for_page(1) is None
        assert idx.get_type_for_page(5) is None

    def test_get_type_for_page_unmapped_entry(self):
        idx = DocumentIndex(
            entries=[
                TOCEntry("Something Unknown", 1, 3, None),
            ],
            source="toc",
        )
        assert idx.get_type_for_page(2) is None

    def test_get_type_for_page_end_page_none_uses_start(self):
        idx = DocumentIndex(
            entries=[
                TOCEntry("Single Page", 5, None, DocumentType.RADIOLOGY),
            ],
            source="toc",
        )
        assert idx.get_type_for_page(5) == DocumentType.RADIOLOGY
        assert idx.get_type_for_page(6) is None

    def test_is_toc_page(self):
        idx = DocumentIndex(
            entries=[],
            source="toc",
            toc_page_numbers=[1, 2],
        )
        assert idx.is_toc_page(1) is True
        assert idx.is_toc_page(2) is True
        assert idx.is_toc_page(3) is False


# ---------------------------------------------------------------------------
# Label -> DocumentType mapping
# ---------------------------------------------------------------------------


class TestLabelToTypeMapping:
    def test_discharge_summary(self):
        assert _map_label_to_type("Discharge Summary") == DocumentType.ED_DISCHARGE_SUMMARY

    def test_operation_report(self):
        assert _map_label_to_type("Operative Report") == DocumentType.ED_OPERATION_REPORT

    def test_radiology(self):
        assert _map_label_to_type("MRI Report") == DocumentType.RADIOLOGY

    def test_outpatient(self):
        assert _map_label_to_type("Outpatient Consultation Notes") == DocumentType.ED_OUTPATIENT_NOTES

    def test_medicolegal(self):
        assert _map_label_to_type("Medicolegal Report") == DocumentType.MEDICOLEGAL_REPORT

    def test_medical_panel(self):
        assert _map_label_to_type("Medical Panel Report") == DocumentType.MEDICAL_PANEL_REPORT

    def test_affidavit(self):
        assert _map_label_to_type("Affidavit of Dr Smith") == DocumentType.AFFIDAVIT

    def test_referral(self):
        assert _map_label_to_type("Referral Letter") == DocumentType.REFERRAL_LETTER

    def test_physiotherapy(self):
        assert _map_label_to_type("Physiotherapy Notes") == DocumentType.CLINICAL_NOTES_ALLIED

    def test_treater_letter(self):
        assert _map_label_to_type("Treater Letter from GP") == DocumentType.TREATER_LETTER_NON_ALLIED

    def test_clinical_notes(self):
        assert _map_label_to_type("Clinical Records") == DocumentType.CLINICAL_NOTES_NON_ALLIED

    def test_specialist(self):
        assert _map_label_to_type("Orthopaedic Specialist") == DocumentType.SPECIALIST_CLINICAL_NOTES

    def test_hospital(self):
        assert _map_label_to_type("Hospital Records") == DocumentType.ED_OTHER

    def test_unknown_label(self):
        assert _map_label_to_type("Random Unknown Label") is None


# ---------------------------------------------------------------------------
# TOC detection
# ---------------------------------------------------------------------------


class TestIsTocPage:
    def test_explicit_table_of_contents_heading(self):
        text = "Table of Contents\n1. Discharge Summary ......... 3\n2. Radiology ......... 8"
        assert is_toc_page(text) is True

    def test_list_of_enclosures_heading(self):
        text = "List of Enclosures\nDischarge Summary - pg 3"
        assert is_toc_page(text) is True

    def test_paginated_index_heading(self):
        text = "Paginated Index\nItem 1 ......... 1"
        assert is_toc_page(text) is True

    def test_dotted_leader_heuristic(self):
        text = (
            "Discharge Summary ......... 3\n"
            "Radiology Report ......... 8\n"
            "Operative Notes ......... 15\n"
        )
        assert is_toc_page(text) is True

    def test_page_reference_heuristic(self):
        text = (
            "Discharge Summary page 3\n"
            "Radiology Report page 8\n"
            "Operative Notes page 15\n"
            "Clinical Notes page 20\n"
            "Medicolegal Report page 35\n"
        )
        assert is_toc_page(text) is True

    def test_normal_page_not_toc(self):
        text = (
            "DISCHARGE SUMMARY\nHospital: Royal Melbourne Hospital\n"
            "Principal diagnosis: Fracture"
        )
        assert is_toc_page(text) is False

    def test_empty_text_not_toc(self):
        assert is_toc_page("") is False
        assert is_toc_page("   ") is False

    def test_few_dotted_leaders_not_enough(self):
        text = "Something ......... 1\nOther thing ......... 2"
        assert is_toc_page(text) is False


class TestFindTocPages:
    def test_finds_toc_page(self):
        pages = [
            _make_page("Table of Contents\n1. Report ......... 3", 1),
            _make_page("DISCHARGE SUMMARY\nHospital text", 2),
        ]
        toc_pages = find_toc_pages(pages)
        assert len(toc_pages) == 1
        assert toc_pages[0].page_number == 1

    def test_no_toc_pages(self):
        pages = [
            _make_page("DISCHARGE SUMMARY\nHospital text", 1),
            _make_page("Radiology Report\nMRI findings", 2),
        ]
        assert find_toc_pages(pages) == []

    def test_multiple_toc_pages(self):
        pages = [
            _make_page("Table of Contents\n1. Report ......... 3", 1),
            _make_page("Contents (continued)\n5. Notes ......... 20\n6. Report ......... 25\n7. Summary ......... 30", 2),
            _make_page("DISCHARGE SUMMARY\nHospital text", 3),
        ]
        toc_pages = find_toc_pages(pages)
        assert len(toc_pages) == 2

    def test_skips_empty_pages(self):
        pages = [
            _make_page("", 1),
            _make_page("   ", 2),
            _make_page("Table of Contents\n1. Report ......... 3", 3),
        ]
        toc_pages = find_toc_pages(pages)
        assert len(toc_pages) == 1
        assert toc_pages[0].page_number == 3


# ---------------------------------------------------------------------------
# TOC entry parsing
# ---------------------------------------------------------------------------


class TestParseTocEntries:
    def test_dotted_leader_entries(self):
        toc_text = (
            "Discharge Summary ......... 3\n"
            "Radiology Report ......... 8\n"
            "Operative Notes ......... 15\n"
        )
        entries = parse_toc_entries(toc_text)
        assert len(entries) == 3
        assert entries[0].label == "Discharge Summary"
        assert entries[0].start_page == 3
        assert entries[1].label == "Radiology Report"
        assert entries[1].start_page == 8
        assert entries[2].label == "Operative Notes"
        assert entries[2].start_page == 15

    def test_entries_sorted_by_start_page(self):
        toc_text = (
            "Notes ......... 20\n"
            "Summary ......... 3\n"
            "Report ......... 10\n"
        )
        entries = parse_toc_entries(toc_text)
        assert entries[0].start_page == 3
        assert entries[1].start_page == 10
        assert entries[2].start_page == 20

    def test_end_pages_inferred(self):
        toc_text = (
            "Discharge Summary ......... 3\n"
            "Radiology Report ......... 8\n"
            "Operative Notes ......... 15\n"
        )
        entries = parse_toc_entries(toc_text)
        assert entries[0].end_page == 7  # 8 - 1
        assert entries[1].end_page == 14  # 15 - 1
        assert entries[2].end_page is None  # last entry

    def test_mapped_types_set(self):
        toc_text = (
            "Discharge Summary ......... 3\n"
            "Radiology Report ......... 8\n"
        )
        entries = parse_toc_entries(toc_text)
        assert entries[0].mapped_type == DocumentType.ED_DISCHARGE_SUMMARY
        assert entries[1].mapped_type == DocumentType.RADIOLOGY

    def test_unknown_labels_have_no_mapped_type(self):
        toc_text = "Random Document ......... 5\n"
        entries = parse_toc_entries(toc_text)
        assert len(entries) == 1
        assert entries[0].mapped_type is None

    def test_page_dash_format(self):
        toc_text = "Discharge Summary - page 3\nRadiology Report - pg 8\n"
        entries = parse_toc_entries(toc_text)
        assert len(entries) == 2
        assert entries[0].start_page == 3
        assert entries[1].start_page == 8

    def test_empty_toc_text(self):
        assert parse_toc_entries("") == []

    def test_short_labels_skipped(self):
        toc_text = "Ab ......... 3\nDischarge Summary ......... 8\n"
        entries = parse_toc_entries(toc_text)
        assert len(entries) == 1
        assert entries[0].label == "Discharge Summary"

    def test_duplicate_page_numbers_skipped(self):
        toc_text = (
            "Discharge Summary ......... 3\n"
            "Another Summary ......... 3\n"
        )
        entries = parse_toc_entries(toc_text)
        assert len(entries) == 1

    def test_zero_page_skipped(self):
        toc_text = "Something ......... 0\nDischarge Summary ......... 3\n"
        entries = parse_toc_entries(toc_text)
        assert len(entries) == 1
        assert entries[0].start_page == 3


# ---------------------------------------------------------------------------
# Build index from TOC
# ---------------------------------------------------------------------------


class TestBuildIndexFromToc:
    def test_builds_index_from_toc_page(self):
        pages = [
            _make_page(
                "Table of Contents\n"
                "Discharge Summary ......... 2\n"
                "Radiology Report ......... 5\n",
                1,
            ),
            _make_page("DISCHARGE SUMMARY\nHospital text", 2),
            _make_page("More discharge text", 3),
            _make_page("More discharge text", 4),
            _make_page("Radiology Report\nMRI findings", 5),
        ]
        index = build_index_from_toc(pages, 5)
        assert index is not None
        assert index.source == "toc"
        assert len(index.entries) == 2
        assert index.toc_page_numbers == [1]
        # Last entry end_page should be set to total_pages
        assert index.entries[-1].end_page == 5

    def test_returns_none_when_no_toc(self):
        pages = [
            _make_page("DISCHARGE SUMMARY\nHospital text", 1),
            _make_page("Radiology Report\nMRI findings", 2),
        ]
        assert build_index_from_toc(pages, 2) is None

    def test_returns_none_when_toc_unparseable(self):
        pages = [
            _make_page("Table of Contents\n(no parseable entries)", 1),
        ]
        assert build_index_from_toc(pages, 1) is None

    def test_toc_page_tracked(self):
        pages = [
            _make_page(
                "Table of Contents\nReport ......... 2\n",
                1,
            ),
            _make_page("Some report text here", 2),
        ]
        index = build_index_from_toc(pages, 2)
        assert index is not None
        assert index.is_toc_page(1) is True
        assert index.is_toc_page(2) is False


# ---------------------------------------------------------------------------
# Generate index from classification results
# ---------------------------------------------------------------------------


class TestGenerateIndexFromClassifications:
    def test_groups_consecutive_same_type(self):
        page_types = [
            (1, DocumentType.ED_DISCHARGE_SUMMARY),
            (2, DocumentType.ED_DISCHARGE_SUMMARY),
            (3, DocumentType.RADIOLOGY),
            (4, DocumentType.RADIOLOGY),
            (5, DocumentType.RADIOLOGY),
        ]
        index = generate_index_from_classifications(page_types)
        assert index.source == "generated"
        assert len(index.entries) == 2
        assert index.entries[0].start_page == 1
        assert index.entries[0].end_page == 2
        assert index.entries[0].mapped_type == DocumentType.ED_DISCHARGE_SUMMARY
        assert index.entries[1].start_page == 3
        assert index.entries[1].end_page == 5
        assert index.entries[1].mapped_type == DocumentType.RADIOLOGY

    def test_single_page_per_type(self):
        page_types = [
            (1, DocumentType.ED_DISCHARGE_SUMMARY),
            (2, DocumentType.RADIOLOGY),
            (3, DocumentType.MEDICOLEGAL_REPORT),
        ]
        index = generate_index_from_classifications(page_types)
        assert len(index.entries) == 3
        assert index.entries[0].start_page == 1
        assert index.entries[0].end_page == 1

    def test_empty_input(self):
        index = generate_index_from_classifications([])
        assert index.source == "generated"
        assert index.entries == []

    def test_all_same_type(self):
        page_types = [
            (1, DocumentType.ED_DISCHARGE_SUMMARY),
            (2, DocumentType.ED_DISCHARGE_SUMMARY),
            (3, DocumentType.ED_DISCHARGE_SUMMARY),
        ]
        index = generate_index_from_classifications(page_types)
        assert len(index.entries) == 1
        assert index.entries[0].start_page == 1
        assert index.entries[0].end_page == 3


# ---------------------------------------------------------------------------
# Integration: TOC-first classification flow via main.py
# ---------------------------------------------------------------------------


class TestTocFirstClassificationFlow:
    """Integration tests for the TOC-guided classification in main.py."""

    def test_toc_page_classified_as_toc(self):
        from unittest.mock import patch
        from pdf_classifier.main import classify_document

        pages = [
            _make_page(
                "Table of Contents\n"
                "Discharge Summary ......... 2\n"
                "Radiology Report ......... 4\n",
                1,
            ),
            _make_page("DISCHARGE SUMMARY\nHospital text", 2),
            _make_page("More hospital discharge data", 3),
            _make_page("Radiology Report\nMRI Findings Impression", 4),
        ]
        with patch("pdf_classifier.main.extract_all_pages", return_value=pages):
            result = classify_document("test.pdf")

        assert result.page_classifications[0].document_type == DocumentType.TOC
        assert result.page_classifications[0].confidence == 1.0

    def test_toc_guided_pages_use_toc_type(self):
        from unittest.mock import patch
        from pdf_classifier.main import classify_document

        pages = [
            _make_page(
                "Table of Contents\n"
                "Discharge Summary ......... 2\n"
                "Radiology Report ......... 4\n",
                1,
            ),
            _make_page("some text here", 2),
            _make_page("some more text", 3),
            _make_page("some radiology text", 4),
        ]
        with patch("pdf_classifier.main.extract_all_pages", return_value=pages):
            result = classify_document("test.pdf")

        # Pages 2-3 should be discharge summary (from TOC)
        assert result.page_classifications[1].document_type == DocumentType.ED_DISCHARGE_SUMMARY
        assert result.page_classifications[2].document_type == DocumentType.ED_DISCHARGE_SUMMARY
        # Page 4 should be radiology (from TOC)
        assert result.page_classifications[3].document_type == DocumentType.RADIOLOGY
        # TOC-guided pages should have the toc_guided flag
        assert "toc_guided" in result.page_classifications[1].flags

    def test_no_toc_falls_back_to_keyword_classifier(self):
        from unittest.mock import patch
        from pdf_classifier.main import classify_document

        pages = [
            _make_page(
                "DISCHARGE SUMMARY\nHospital: Royal Melbourne Hospital\n"
                "Principal diagnosis: Fracture\nFollow up: 6 weeks",
                1,
            ),
            _make_page(
                "Radiology Report\nMRI lumbar spine\n"
                "Findings: Disc protrusion\nImpression: L4/5 disc herniation",
                2,
            ),
        ]
        with patch("pdf_classifier.main.extract_all_pages", return_value=pages):
            result = classify_document("test.pdf")

        assert result.page_classifications[0].document_type == DocumentType.ED_DISCHARGE_SUMMARY
        assert result.page_classifications[1].document_type in (
            DocumentType.RADIOLOGY, DocumentType.ED_RADIOLOGY,
        )
        # No toc_guided flag when no TOC is used
        assert "toc_guided" not in result.page_classifications[0].flags

    def test_toc_guided_confidence_value(self):
        from unittest.mock import patch
        from pdf_classifier.main import classify_document, _TOC_GUIDED_CONFIDENCE

        pages = [
            _make_page(
                "Table of Contents\n"
                "Discharge Summary ......... 2\n",
                1,
            ),
            _make_page("some text here", 2),
        ]
        with patch("pdf_classifier.main.extract_all_pages", return_value=pages):
            result = classify_document("test.pdf")

        assert result.page_classifications[1].confidence == _TOC_GUIDED_CONFIDENCE
