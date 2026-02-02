"""Tests for data models."""

import json
import pytest
from pathlib import Path

from pdf_classifier.models import DocumentResult, PageClassification
from pdf_classifier.taxonomy import (
    DocumentCategory,
    DocumentType,
    PriorityLevel,
    SourceType,
)


def _make_classification(
    page_number: int = 1,
    doc_type: DocumentType = DocumentType.ED_DISCHARGE_SUMMARY,
    source: SourceType = SourceType.PRIMARY,
    priority: PriorityLevel = PriorityLevel.HIGH,
    confidence: float = 0.8,
    needs_review: bool = False,
) -> PageClassification:
    return PageClassification(
        page_number=page_number,
        document_type=doc_type,
        source_type=source,
        priority_level=priority,
        confidence=confidence,
        extracted_text="Sample text",
        classification_reasoning="Test reasoning",
        needs_manual_review=needs_review,
    )


class TestPageClassification:
    def test_category_property(self):
        pc = _make_classification(doc_type=DocumentType.ED_DISCHARGE_SUMMARY)
        assert pc.category == DocumentCategory.HOSPITAL_ED

    def test_category_for_legal_type(self):
        pc = _make_classification(doc_type=DocumentType.MEDICOLEGAL_REPORT)
        assert pc.category == DocumentCategory.LEGAL_ASSESSMENT

    def test_to_dict_contains_required_fields(self):
        pc = _make_classification()
        d = pc.to_dict()
        assert "page_number" in d
        assert "document_type" in d
        assert "category" in d
        assert "source_type" in d
        assert "priority_level" in d
        assert "confidence" in d
        assert "needs_manual_review" in d

    def test_to_dict_values(self):
        pc = _make_classification(
            page_number=5,
            doc_type=DocumentType.RADIOLOGY,
            source=SourceType.PRIMARY,
            priority=PriorityLevel.HIGH,
            confidence=0.75,
        )
        d = pc.to_dict()
        assert d["page_number"] == 5
        assert d["document_type"] == "Radiology"
        assert d["source_type"] == "Primary"
        assert d["priority_level"] == "High"
        assert d["confidence"] == 0.75


class TestDocumentResult:
    def _make_result(self) -> DocumentResult:
        result = DocumentResult(file_path="test.pdf", total_pages=4)
        result.page_classifications = [
            _make_classification(1, DocumentType.ED_DISCHARGE_SUMMARY),
            _make_classification(2, DocumentType.RADIOLOGY),
            _make_classification(3, DocumentType.MEDICOLEGAL_REPORT,
                                 SourceType.SECONDARY, PriorityLevel.LOW),
            _make_classification(4, DocumentType.ED_DISCHARGE_SUMMARY,
                                 needs_review=True),
        ]
        return result

    def test_classification_summary(self):
        result = self._make_result()
        summary = result.classification_summary
        assert summary["ED/Hospital presentation (discharge summary)"] == 2
        assert summary["Radiology"] == 1
        assert summary["Medicolegal report"] == 1

    def test_source_type_summary(self):
        result = self._make_result()
        summary = result.source_type_summary
        assert summary["Primary"] == 3
        assert summary["Secondary"] == 1

    def test_priority_summary(self):
        result = self._make_result()
        summary = result.priority_summary
        assert summary["High"] == 3
        assert summary["Low"] == 1

    def test_pages_needing_review(self):
        result = self._make_result()
        review = result.pages_needing_review
        assert len(review) == 1
        assert review[0].page_number == 4

    def test_pages_by_type(self):
        result = self._make_result()
        discharge_pages = result.pages_by_type(
            DocumentType.ED_DISCHARGE_SUMMARY
        )
        assert len(discharge_pages) == 2

    def test_pages_by_category(self):
        result = self._make_result()
        hospital_pages = result.pages_by_category(
            DocumentCategory.HOSPITAL_ED
        )
        assert len(hospital_pages) == 2

    def test_pages_by_source(self):
        result = self._make_result()
        primary = result.pages_by_source(SourceType.PRIMARY)
        assert len(primary) == 3

    def test_pages_by_priority(self):
        result = self._make_result()
        high = result.pages_by_priority(PriorityLevel.HIGH)
        assert len(high) == 3

    def test_to_dict(self):
        result = self._make_result()
        d = result.to_dict()
        assert d["file_path"] == "test.pdf"
        assert d["total_pages"] == 4
        assert len(d["pages"]) == 4
        assert "classification_summary" in d
        assert "source_type_summary" in d
        assert "priority_summary" in d

    def test_to_json(self):
        result = self._make_result()
        j = result.to_json()
        parsed = json.loads(j)
        assert parsed["total_pages"] == 4
        assert len(parsed["pages"]) == 4

    def test_save_json(self, tmp_path: Path):
        result = self._make_result()
        output_file = tmp_path / "output" / "results.json"
        result.save_json(output_file)
        assert output_file.exists()
        parsed = json.loads(output_file.read_text())
        assert parsed["total_pages"] == 4
