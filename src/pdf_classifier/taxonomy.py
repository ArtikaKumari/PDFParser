"""Classification taxonomy definitions for medicolegal document pages."""

from enum import Enum


class DocumentCategory(str, Enum):
    """Top-level document category groupings."""

    HOSPITAL_ED = "Hospital/ED Presentations"
    STANDALONE_MEDICAL = "Standalone Medical Documents"
    CORRESPONDENCE = "Correspondence"
    CLINICAL_DOCUMENTATION = "Clinical Documentation"
    LEGAL_ASSESSMENT = "Legal/Assessment Documents"
    FALLBACK = "Fallback"


class DocumentType(str, Enum):
    """Specific document type classifications."""

    # Hospital/ED Presentations
    ED_DISCHARGE_SUMMARY = "ED/Hospital presentation (discharge summary)"
    ED_OPERATION_REPORT = "ED/Hospital presentation (operation report)"
    ED_RADIOLOGY = "ED/Hospital presentation (radiology)"
    ED_OUTPATIENT_NOTES = "ED/Hospital presentation (outpatient consultation notes)"
    ED_OTHER = "ED/Hospital presentation (other)"

    # Standalone Medical Documents
    RADIOLOGY = "Radiology"
    OPERATION_REPORT = "Operation report"

    # Correspondence
    TREATER_LETTER_NON_ALLIED = "Treater letter (non-allied health)"
    TREATER_LETTER_ALLIED = "Treater letter (allied health)"
    REFERRAL_LETTER = "Referral letter"

    # Clinical Documentation
    CLINICAL_NOTES_NON_ALLIED = "Clinical notes (non-allied health)"
    CLINICAL_NOTES_ALLIED = "Clinical notes (allied health)"
    SPECIALIST_CLINICAL_NOTES = "Specialist clinical notes"

    # Legal/Assessment Documents
    MEDICOLEGAL_REPORT = "Medicolegal report"
    MEDICAL_PANEL_REPORT = "Medical Panel report"
    AFFIDAVIT = "Affidavit"

    # Fallback
    OTHER = "Other"


class SourceType(str, Enum):
    """Whether a document is a primary or secondary source."""

    PRIMARY = "Primary"
    SECONDARY = "Secondary"


class PriorityLevel(str, Enum):
    """Importance/priority level for documents."""

    HIGH = "High"
    MID = "Mid"
    LOW = "Low"


# Mapping from DocumentType to its parent category
DOCUMENT_TYPE_TO_CATEGORY: dict[DocumentType, DocumentCategory] = {
    DocumentType.ED_DISCHARGE_SUMMARY: DocumentCategory.HOSPITAL_ED,
    DocumentType.ED_OPERATION_REPORT: DocumentCategory.HOSPITAL_ED,
    DocumentType.ED_RADIOLOGY: DocumentCategory.HOSPITAL_ED,
    DocumentType.ED_OUTPATIENT_NOTES: DocumentCategory.HOSPITAL_ED,
    DocumentType.ED_OTHER: DocumentCategory.HOSPITAL_ED,
    DocumentType.RADIOLOGY: DocumentCategory.STANDALONE_MEDICAL,
    DocumentType.OPERATION_REPORT: DocumentCategory.STANDALONE_MEDICAL,
    DocumentType.TREATER_LETTER_NON_ALLIED: DocumentCategory.CORRESPONDENCE,
    DocumentType.TREATER_LETTER_ALLIED: DocumentCategory.CORRESPONDENCE,
    DocumentType.REFERRAL_LETTER: DocumentCategory.CORRESPONDENCE,
    DocumentType.CLINICAL_NOTES_NON_ALLIED: DocumentCategory.CLINICAL_DOCUMENTATION,
    DocumentType.CLINICAL_NOTES_ALLIED: DocumentCategory.CLINICAL_DOCUMENTATION,
    DocumentType.SPECIALIST_CLINICAL_NOTES: DocumentCategory.CLINICAL_DOCUMENTATION,
    DocumentType.MEDICOLEGAL_REPORT: DocumentCategory.LEGAL_ASSESSMENT,
    DocumentType.MEDICAL_PANEL_REPORT: DocumentCategory.LEGAL_ASSESSMENT,
    DocumentType.AFFIDAVIT: DocumentCategory.LEGAL_ASSESSMENT,
    DocumentType.OTHER: DocumentCategory.FALLBACK,
}

# Default source type for document types that are always one type
ALWAYS_SECONDARY_TYPES: set[DocumentType] = {
    DocumentType.MEDICOLEGAL_REPORT,
    DocumentType.MEDICAL_PANEL_REPORT,
    DocumentType.AFFIDAVIT,
}

# Default primary source types (may be overridden by context analysis)
DEFAULT_PRIMARY_TYPES: set[DocumentType] = {
    DocumentType.ED_DISCHARGE_SUMMARY,
    DocumentType.ED_OPERATION_REPORT,
    DocumentType.ED_RADIOLOGY,
    DocumentType.ED_OUTPATIENT_NOTES,
    DocumentType.ED_OTHER,
    DocumentType.RADIOLOGY,
    DocumentType.OPERATION_REPORT,
    DocumentType.TREATER_LETTER_NON_ALLIED,
    DocumentType.TREATER_LETTER_ALLIED,
    DocumentType.REFERRAL_LETTER,
    DocumentType.CLINICAL_NOTES_NON_ALLIED,
    DocumentType.CLINICAL_NOTES_ALLIED,
    DocumentType.SPECIALIST_CLINICAL_NOTES,
}

# Priority mappings for primary sources
PRIMARY_SOURCE_PRIORITY: dict[DocumentType, PriorityLevel] = {
    DocumentType.ED_DISCHARGE_SUMMARY: PriorityLevel.HIGH,
    DocumentType.ED_OPERATION_REPORT: PriorityLevel.HIGH,
    DocumentType.ED_RADIOLOGY: PriorityLevel.HIGH,
    DocumentType.ED_OUTPATIENT_NOTES: PriorityLevel.MID,
    DocumentType.ED_OTHER: PriorityLevel.MID,
    DocumentType.RADIOLOGY: PriorityLevel.HIGH,
    DocumentType.OPERATION_REPORT: PriorityLevel.HIGH,
    DocumentType.TREATER_LETTER_NON_ALLIED: PriorityLevel.HIGH,
    DocumentType.TREATER_LETTER_ALLIED: PriorityLevel.MID,
    DocumentType.REFERRAL_LETTER: PriorityLevel.MID,
    DocumentType.CLINICAL_NOTES_NON_ALLIED: PriorityLevel.MID,
    DocumentType.CLINICAL_NOTES_ALLIED: PriorityLevel.MID,
    DocumentType.SPECIALIST_CLINICAL_NOTES: PriorityLevel.MID,
}

# Priority mappings for secondary sources
SECONDARY_SOURCE_PRIORITY: dict[DocumentType, PriorityLevel] = {
    DocumentType.MEDICAL_PANEL_REPORT: PriorityLevel.HIGH,
    DocumentType.AFFIDAVIT: PriorityLevel.MID,
    DocumentType.MEDICOLEGAL_REPORT: PriorityLevel.LOW,
}

# Allied health professions for classification
ALLIED_HEALTH_PROFESSIONS: list[str] = [
    "physiotherapist",
    "physiotherapy",
    "physio",
    "occupational therapist",
    "occupational therapy",
    "speech pathologist",
    "speech therapy",
    "dietitian",
    "dietetics",
    "podiatrist",
    "podiatry",
    "osteopath",
    "osteopathy",
    "chiropractor",
    "chiropractic",
    "psychologist",
    "psychology",
    "social worker",
    "social work",
    "exercise physiologist",
    "exercise physiology",
    "audiologist",
    "audiology",
    "optometrist",
    "optometry",
    "orthoptist",
    "orthotics",
    "prosthetics",
    "counsellor",
    "counselor",
    "myotherapist",
    "myotherapy",
    "remedial massage",
    "acupuncturist",
    "naturopath",
]

# Legal addressees that indicate medicolegal context
LEGAL_ADDRESSEES: list[str] = [
    "court",
    "tribunal",
    "magistrate",
    "judge",
    "solicitor",
    "barrister",
    "lawyer",
    "legal",
    "law firm",
    "attorneys",
    "insurance",
    "insurer",
    "tac",
    "transport accident commission",
    "workcover",
    "worksafe",
    "comcare",
    "workers compensation",
    "worker's compensation",
    "claims",
    "liability",
]
