"""Priority/importance level assignment for classified pages."""

from __future__ import annotations

from pdf_classifier.taxonomy import (
    DocumentType,
    PriorityLevel,
    SourceType,
    PRIMARY_SOURCE_PRIORITY,
    SECONDARY_SOURCE_PRIORITY,
)


def assign_priority(
    document_type: DocumentType, source_type: SourceType
) -> PriorityLevel:
    """Assign an importance/priority level to a classified page.

    Priority is determined by both the document type and its source type
    classification, following the defined priority mappings.

    Args:
        document_type: The classified document type.
        source_type: Whether this is a primary or secondary source.

    Returns:
        The assigned PriorityLevel.
    """
    if source_type == SourceType.SECONDARY:
        # Check secondary-specific priority first
        if document_type in SECONDARY_SOURCE_PRIORITY:
            return SECONDARY_SOURCE_PRIORITY[document_type]
        # For primary-default types that were reclassified as secondary,
        # drop one priority level from their primary priority
        if document_type in PRIMARY_SOURCE_PRIORITY:
            primary_priority = PRIMARY_SOURCE_PRIORITY[document_type]
            if primary_priority == PriorityLevel.HIGH:
                return PriorityLevel.MID
            return PriorityLevel.LOW
        return PriorityLevel.LOW

    # Primary source
    if document_type in PRIMARY_SOURCE_PRIORITY:
        return PRIMARY_SOURCE_PRIORITY[document_type]

    # Fallback for unclassified types
    return PriorityLevel.LOW
