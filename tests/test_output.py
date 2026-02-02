"""Tests for output formatting and export functions."""

import csv
import json
import pytest
from pathlib import Path

from pdf_classifier.models import DocumentResult, PageClassification
from pdf_classifier.output import (
    export_csv,
    export_grouped_by_category,
    export_json,
    format_page_summary,
    format_report,
)
from pdf_classifier.taxonomy import (
    DocumentCategory,
    DocumentType,
    PriorityLevel,
    SourceType,
)


def _make_result() -> DocumentResult:
    result = DocumentResult(file_path="test.pdf", total_pages=3)
    result.page_classifications = [
        PageClassification(
            page_number=1,
            document_type=DocumentType.ED_DISCHARGE_SUMMARY,
            source_type=SourceType.PRIMARY,
            priority_level=PriorityLevel.HIGH,
            confidence=0.85,
            extracted_text="Discharge text",
            classification_reasoning="Test reasoning",
        ),
        PageClassification(
            page_number=2,
            document_type=DocumentType.RADIOLOGY,
            source_type=SourceType.PRIMARY,
            priority_level=PriorityLevel.HIGH,
            confidence=0.72,
            extracted_text="Radiology text",
            classification_reasoning="Test reasoning",
        ),
        PageClassification(
            page_number=3,
            document_type=DocumentType.MEDICOLEGAL_REPORT,
            source_type=SourceType.SECONDARY,
            priority_level=PriorityLevel.LOW,
            confidence=0.60,
            extracted_text="Medicolegal text",
            classification_reasoning="Test reasoning",
            needs_manual_review=True,
        ),
    ]
    return result


class TestFormatPageSummary:
    def test_includes_page_number(self):
        pc = _make_result().page_classifications[0]
        line = format_page_summary(pc)
        assert "1" in line

    def test_includes_document_type(self):
        pc = _make_result().page_classifications[0]
        line = format_page_summary(pc)
        assert "discharge summary" in line.lower()

    def test_includes_source_type(self):
        pc = _make_result().page_classifications[0]
        line = format_page_summary(pc)
        assert "Primary" in line

    def test_includes_review_flag(self):
        pc = _make_result().page_classifications[2]
        line = format_page_summary(pc)
        assert "NEEDS REVIEW" in line

    def test_no_review_flag_when_not_needed(self):
        pc = _make_result().page_classifications[0]
        line = format_page_summary(pc)
        assert "NEEDS REVIEW" not in line


class TestFormatReport:
    def test_report_contains_header(self):
        report = format_report(_make_result())
        assert "PDF DOCUMENT CLASSIFICATION REPORT" in report

    def test_report_contains_file_path(self):
        report = format_report(_make_result())
        assert "test.pdf" in report

    def test_report_contains_total_pages(self):
        report = format_report(_make_result())
        assert "3" in report

    def test_report_contains_classification_summary(self):
        report = format_report(_make_result())
        assert "Classification Summary" in report

    def test_report_contains_source_type_summary(self):
        report = format_report(_make_result())
        assert "Source Type Summary" in report

    def test_report_contains_priority_summary(self):
        report = format_report(_make_result())
        assert "Priority Summary" in report

    def test_report_contains_page_details(self):
        report = format_report(_make_result())
        assert "Page-by-Page Classification" in report

    def test_report_contains_manual_review_section(self):
        report = format_report(_make_result())
        assert "Manual Review" in report


class TestExportCsv:
    def test_csv_file_created(self, tmp_path: Path):
        output = tmp_path / "results.csv"
        export_csv(_make_result(), output)
        assert output.exists()

    def test_csv_has_correct_headers(self, tmp_path: Path):
        output = tmp_path / "results.csv"
        export_csv(_make_result(), output)
        with open(output, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        assert "page_number" in headers
        assert "document_type" in headers
        assert "category" in headers
        assert "source_type" in headers
        assert "priority_level" in headers
        assert "confidence" in headers

    def test_csv_has_correct_row_count(self, tmp_path: Path):
        output = tmp_path / "results.csv"
        export_csv(_make_result(), output)
        with open(output, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3

    def test_csv_creates_parent_dirs(self, tmp_path: Path):
        output = tmp_path / "subdir" / "nested" / "results.csv"
        export_csv(_make_result(), output)
        assert output.exists()


class TestExportJson:
    def test_json_file_created(self, tmp_path: Path):
        output = tmp_path / "results.json"
        export_json(_make_result(), output)
        assert output.exists()

    def test_json_is_valid(self, tmp_path: Path):
        output = tmp_path / "results.json"
        export_json(_make_result(), output)
        data = json.loads(output.read_text())
        assert data["total_pages"] == 3
        assert len(data["pages"]) == 3

    def test_json_creates_parent_dirs(self, tmp_path: Path):
        output = tmp_path / "subdir" / "results.json"
        export_json(_make_result(), output)
        assert output.exists()


class TestExportGroupedByCategory:
    def test_groups_by_category(self):
        grouped = export_grouped_by_category(_make_result())
        assert "Hospital/ED Presentations" in grouped
        assert "Standalone Medical Documents" in grouped
        assert "Legal/Assessment Documents" in grouped

    def test_correct_page_counts_per_category(self):
        grouped = export_grouped_by_category(_make_result())
        assert len(grouped["Hospital/ED Presentations"]) == 1
        assert len(grouped["Standalone Medical Documents"]) == 1
        assert len(grouped["Legal/Assessment Documents"]) == 1

    def test_empty_categories_excluded(self):
        grouped = export_grouped_by_category(_make_result())
        assert "Correspondence" not in grouped
        assert "Clinical Documentation" not in grouped
