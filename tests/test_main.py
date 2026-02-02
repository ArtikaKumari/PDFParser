"""Tests for the main application orchestrator and CLI."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from pdf_classifier.extractor import PageContent
from pdf_classifier.main import classify_document, main, _classify_single_page
from pdf_classifier.taxonomy import DocumentType, SourceType, PriorityLevel


def _make_page_content(
    text: str, page_number: int = 1
) -> PageContent:
    return PageContent(
        page_number=page_number,
        text=text,
        extraction_method="text",
        has_images=False,
        word_count=len(text.split()),
    )


class TestClassifySinglePage:
    def test_discharge_page_full_pipeline(self):
        page = _make_page_content(
            "DISCHARGE SUMMARY\nHospital: Royal Melbourne Hospital\n"
            "Principal diagnosis: Fracture\nFollow up: 6 weeks"
        )
        result = _classify_single_page(page)
        assert result.document_type == DocumentType.ED_DISCHARGE_SUMMARY
        assert result.source_type == SourceType.PRIMARY
        assert result.priority_level == PriorityLevel.HIGH
        assert result.confidence > 0.0

    def test_medicolegal_page_full_pipeline(self):
        page = _make_page_content(
            "MEDICOLEGAL REPORT\n"
            "Independent Medical Examination\n"
            "Prepared at the request of: Slater & Gordon Lawyers\n"
            "Instructing solicitor: Ms A. Lawyer\n"
            "Whole person impairment: 10% WPI"
        )
        result = _classify_single_page(page)
        assert result.document_type == DocumentType.MEDICOLEGAL_REPORT
        assert result.source_type == SourceType.SECONDARY
        assert result.priority_level == PriorityLevel.LOW

    def test_empty_page_flagged_for_review(self):
        page = _make_page_content("")
        page.extraction_method = "empty"
        result = _classify_single_page(page)
        assert result.needs_manual_review
        assert "no_text_content" in result.flags

    def test_ocr_page_flagged(self):
        page = _make_page_content("Some OCR text from scanned document")
        page.extraction_method = "ocr"
        result = _classify_single_page(page)
        assert "ocr_extracted" in result.flags

    def test_short_content_flagged(self):
        page = _make_page_content("Brief text")
        page.word_count = 2
        result = _classify_single_page(page)
        assert "very_short_content" in result.flags

    def test_reasoning_contains_type_and_source(self):
        page = _make_page_content(
            "Radiology Report\nMRI lumbar spine\n"
            "Findings: Disc protrusion\nImpression: L4/5 disc herniation"
        )
        result = _classify_single_page(page)
        assert "Type:" in result.classification_reasoning
        assert "Source:" in result.classification_reasoning


class TestClassifyDocument:
    @patch("pdf_classifier.main.extract_all_pages")
    def test_processes_all_pages(self, mock_extract):
        mock_extract.return_value = [
            _make_page_content("DISCHARGE SUMMARY Hospital discharge", 1),
            _make_page_content(
                "Radiology Report MRI Findings Impression radiologist", 2
            ),
        ]
        result = classify_document("dummy.pdf")
        assert result.total_pages == 2
        assert len(result.page_classifications) == 2

    @patch("pdf_classifier.main.extract_all_pages")
    def test_handles_classification_errors_gracefully(self, mock_extract):
        mock_extract.return_value = [
            _make_page_content("Normal discharge summary hospital text", 1),
        ]
        # Patch classify_page to raise an exception
        with patch("pdf_classifier.main.classify_page", side_effect=RuntimeError("test")):
            result = classify_document("dummy.pdf")
            # Should still have a result for the page (fallback)
            assert len(result.page_classifications) == 1
            assert result.page_classifications[0].needs_manual_review
            assert len(result.processing_errors) == 1

    @patch("pdf_classifier.main.extract_all_pages")
    def test_result_file_path(self, mock_extract):
        mock_extract.return_value = []
        result = classify_document("my_document.pdf")
        assert result.file_path == "my_document.pdf"


class TestCLI:
    def test_missing_file_returns_1(self):
        exit_code = main(["nonexistent.pdf"])
        assert exit_code == 1

    def test_non_pdf_file_returns_1(self, tmp_path: Path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("hello")
        exit_code = main([str(txt_file)])
        assert exit_code == 1

    @patch("pdf_classifier.main.classify_document")
    def test_classify_only_mode(self, mock_classify, tmp_path: Path):
        from pdf_classifier.models import DocumentResult
        mock_classify.return_value = DocumentResult(
            file_path="test.pdf", total_pages=0
        )
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")
        exit_code = main([str(pdf_file), "--classify-only"])
        assert exit_code == 0
        mock_classify.assert_called_once()

    @patch("pdf_classifier.main.process_document")
    def test_full_process_mode(self, mock_process, tmp_path: Path):
        from pdf_classifier.models import DocumentResult
        mock_process.return_value = (
            DocumentResult(file_path="test.pdf", total_pages=0),
            [],
        )
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")
        exit_code = main([str(pdf_file)])
        assert exit_code == 0
        mock_process.assert_called_once()

    @patch("pdf_classifier.main.classify_document")
    def test_json_output_to_dir(self, mock_classify, tmp_path: Path):
        from pdf_classifier.models import DocumentResult
        mock_classify.return_value = DocumentResult(
            file_path="test.pdf", total_pages=0
        )
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")
        out_dir = tmp_path / "output"
        exit_code = main([
            str(pdf_file),
            "--classify-only",
            "--output-dir", str(out_dir),
            "--format", "json",
        ])
        assert exit_code == 0
        assert (out_dir / "test_results.json").exists()
