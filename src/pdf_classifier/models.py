"""Data models for PDF classification results."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from pdf_classifier.taxonomy import (
    DocumentCategory,
    DocumentType,
    PriorityLevel,
    SourceType,
    DOCUMENT_TYPE_TO_CATEGORY,
)

logger = logging.getLogger(__name__)


@dataclass
class PageClassification:
    """Classification result for a single PDF page."""

    page_number: int
    document_type: DocumentType
    source_type: SourceType
    priority_level: PriorityLevel
    confidence: float
    extracted_text: str = ""
    classification_reasoning: str = ""
    flags: list[str] = field(default_factory=list)
    needs_manual_review: bool = False

    @property
    def category(self) -> DocumentCategory:
        """Get the parent category for this document type."""
        return DOCUMENT_TYPE_TO_CATEGORY[self.document_type]

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for serialization."""
        return {
            "page_number": self.page_number,
            "document_type": self.document_type.value,
            "category": self.category.value,
            "source_type": self.source_type.value,
            "priority_level": self.priority_level.value,
            "confidence": round(self.confidence, 3),
            "classification_reasoning": self.classification_reasoning,
            "flags": self.flags,
            "needs_manual_review": self.needs_manual_review,
        }


@dataclass
class DocumentResult:
    """Complete classification result for an entire PDF document."""

    file_path: str
    total_pages: int
    page_classifications: list[PageClassification] = field(default_factory=list)
    processing_errors: list[str] = field(default_factory=list)

    @property
    def pages_needing_review(self) -> list[PageClassification]:
        """Return pages flagged for manual review."""
        return [p for p in self.page_classifications if p.needs_manual_review]

    @property
    def classification_summary(self) -> dict[str, int]:
        """Count of pages per document type."""
        summary: dict[str, int] = {}
        for page in self.page_classifications:
            key = page.document_type.value
            summary[key] = summary.get(key, 0) + 1
        return summary

    @property
    def source_type_summary(self) -> dict[str, int]:
        """Count of pages per source type."""
        summary: dict[str, int] = {}
        for page in self.page_classifications:
            key = page.source_type.value
            summary[key] = summary.get(key, 0) + 1
        return summary

    @property
    def priority_summary(self) -> dict[str, int]:
        """Count of pages per priority level."""
        summary: dict[str, int] = {}
        for page in self.page_classifications:
            key = page.priority_level.value
            summary[key] = summary.get(key, 0) + 1
        return summary

    def pages_by_type(self, doc_type: DocumentType) -> list[PageClassification]:
        """Get all pages matching a specific document type."""
        return [
            p for p in self.page_classifications if p.document_type == doc_type
        ]

    def pages_by_category(
        self, category: DocumentCategory
    ) -> list[PageClassification]:
        """Get all pages matching a specific category."""
        return [
            p for p in self.page_classifications if p.category == category
        ]

    def pages_by_source(
        self, source_type: SourceType
    ) -> list[PageClassification]:
        """Get all pages matching a specific source type."""
        return [
            p for p in self.page_classifications if p.source_type == source_type
        ]

    def pages_by_priority(
        self, priority: PriorityLevel
    ) -> list[PageClassification]:
        """Get all pages matching a specific priority level."""
        return [
            p
            for p in self.page_classifications
            if p.priority_level == priority
        ]

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for serialization."""
        return {
            "file_path": self.file_path,
            "total_pages": self.total_pages,
            "classification_summary": self.classification_summary,
            "source_type_summary": self.source_type_summary,
            "priority_summary": self.priority_summary,
            "pages": [p.to_dict() for p in self.page_classifications],
            "processing_errors": self.processing_errors,
            "pages_needing_review": [
                p.page_number for p in self.pages_needing_review
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, output_path: str | Path) -> None:
        """Save classification results to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        logger.info("Classification results saved to %s", path)
