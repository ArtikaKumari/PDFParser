"""Page-level document classification engine.

Implements rule-based classification using keyword patterns, structural
indicators, and the medicolegal identification rules defined in the taxonomy.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from pdf_classifier.extractor import PageContent
from pdf_classifier.taxonomy import (
    ALLIED_HEALTH_PROFESSIONS,
    DocumentType,
    LEGAL_ADDRESSEES,
)

logger = logging.getLogger(__name__)


@dataclass
class _ClassificationScore:
    """Internal scoring result for a candidate classification."""

    document_type: DocumentType
    score: float
    reasons: list[str] = field(default_factory=list)


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace for pattern matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _contains_any(text: str, keywords: list[str]) -> list[str]:
    """Return which keywords appear in the text."""
    return [kw for kw in keywords if kw in text]


def _is_allied_health(text: str) -> bool:
    """Check if the text references an allied health profession."""
    return bool(_contains_any(text, ALLIED_HEALTH_PROFESSIONS))


def _has_legal_addressee(text: str) -> bool:
    """Check if the document is addressed to a legal/insurance entity."""
    return bool(_contains_any(text, LEGAL_ADDRESSEES))


def _has_hospital_context(text: str) -> bool:
    """Check for hospital/ED presentation context markers."""
    markers = [
        "emergency department",
        "emergency dept",
        "hospital",
        "admitted",
        "admission",
        "inpatient",
        "ward",
        "presenting complaint",
        "triage",
        "discharge",
        "length of stay",
    ]
    return bool(_contains_any(text, markers))


def _has_non_treating_practitioner_indicators(text: str) -> bool:
    """Check for indicators that the author is NOT the treating practitioner.

    Medicolegal reports are written by practitioners not involved in treatment.
    Common indicators include phrases like 'independent medical examination',
    'at the request of', 'medicolegal assessment', etc.
    """
    indicators = [
        "independent medical examination",
        "independent medical examiner",
        "independent medical assessment",
        "medicolegal assessment",
        "medicolegal examination",
        "medico-legal assessment",
        "medico-legal examination",
        "medico-legal report",
        "medicolegal report",
        "medicolegal opinion",
        "medico-legal opinion",
        "at the request of",
        "instructed by",
        "instructing solicitor",
        "i was asked to examine",
        "i was requested to examine",
        "i was asked to assess",
        "i was requested to assess",
        "referred for assessment",
        "referred for examination",
        "independent assessment",
        "for the purpose of",
        "permanent impairment",
        "whole person impairment",
        "impairment assessment",
        "impairment rating",
    ]
    return bool(_contains_any(text, indicators))


# ---------------------------------------------------------------------------
# Individual scoring functions for each document type
# ---------------------------------------------------------------------------


def _score_ed_discharge(text: str) -> _ClassificationScore:
    cs = _ClassificationScore(DocumentType.ED_DISCHARGE_SUMMARY, 0.0)
    discharge_kw = [
        "discharge summary",
        "discharge letter",
        "discharged",
        "discharge diagnosis",
        "discharge plan",
        "discharge medication",
    ]
    hits = _contains_any(text, discharge_kw)
    if hits:
        cs.score += 0.4
        cs.reasons.append(f"discharge keywords: {hits}")
    if _has_hospital_context(text):
        cs.score += 0.3
        cs.reasons.append("hospital context present")
    summary_kw = [
        "principal diagnosis",
        "procedures performed",
        "follow up",
        "follow-up",
        "condition on discharge",
    ]
    hits = _contains_any(text, summary_kw)
    if hits:
        cs.score += 0.2
        cs.reasons.append(f"summary structure keywords: {hits}")
    return cs


def _score_ed_operation(text: str) -> _ClassificationScore:
    cs = _ClassificationScore(DocumentType.ED_OPERATION_REPORT, 0.0)
    op_kw = [
        "operation report",
        "operative report",
        "surgical report",
        "procedure report",
        "surgeon",
        "anaesthetist",
        "anesthetist",
        "anaesthesia",
        "anesthesia",
        "operative findings",
        "operative procedure",
    ]
    hits = _contains_any(text, op_kw)
    if hits:
        cs.score += 0.4
        cs.reasons.append(f"operation keywords: {hits}")
    if _has_hospital_context(text):
        cs.score += 0.3
        cs.reasons.append("hospital context present")
    detail_kw = [
        "incision",
        "excision",
        "suture",
        "tourniquet",
        "specimen",
        "post-operative",
        "postoperative",
        "pre-operative",
        "preoperative",
    ]
    hits = _contains_any(text, detail_kw)
    if hits:
        cs.score += 0.2
        cs.reasons.append(f"surgical detail keywords: {hits}")
    return cs


def _score_ed_radiology(text: str) -> _ClassificationScore:
    cs = _ClassificationScore(DocumentType.ED_RADIOLOGY, 0.0)
    rad_kw = [
        "radiology report",
        "imaging report",
        "x-ray",
        "xray",
        "ct scan",
        "mri",
        "ultrasound",
        "radiograph",
        "fluoroscopy",
    ]
    hits = _contains_any(text, rad_kw)
    if hits:
        cs.score += 0.4
        cs.reasons.append(f"radiology keywords: {hits}")
    if _has_hospital_context(text):
        cs.score += 0.3
        cs.reasons.append("hospital context present")
    findings_kw = [
        "impression",
        "findings",
        "clinical indication",
        "comparison",
        "technique",
        "radiologist",
    ]
    hits = _contains_any(text, findings_kw)
    if hits:
        cs.score += 0.2
        cs.reasons.append(f"findings structure keywords: {hits}")
    # Penalize if this looks like clinical notes that merely reference imaging
    notes_context = [
        "clinical notes",
        "presenting complaint",
        "refer for",
        "plan:",
        "assessment:",
    ]
    if _contains_any(text, notes_context):
        cs.score -= 0.3
        cs.reasons.append("clinical notes context penalty")
    return cs


def _score_ed_outpatient(text: str) -> _ClassificationScore:
    cs = _ClassificationScore(DocumentType.ED_OUTPATIENT_NOTES, 0.0)
    out_kw = [
        "outpatient",
        "out-patient",
        "clinic review",
        "clinic visit",
        "clinic appointment",
        "follow up clinic",
        "follow-up clinic",
        "review appointment",
    ]
    hits = _contains_any(text, out_kw)
    if hits:
        cs.score += 0.4
        cs.reasons.append(f"outpatient keywords: {hits}")
    if _has_hospital_context(text):
        cs.score += 0.3
        cs.reasons.append("hospital context present")
    consult_kw = [
        "consultation",
        "consultation notes",
        "attending",
        "registrar",
        "consultant",
    ]
    hits = _contains_any(text, consult_kw)
    if hits:
        cs.score += 0.15
        cs.reasons.append(f"consultation keywords: {hits}")
    return cs


def _score_ed_other(text: str) -> _ClassificationScore:
    cs = _ClassificationScore(DocumentType.ED_OTHER, 0.0)
    if _has_hospital_context(text):
        cs.score += 0.4
        cs.reasons.append("hospital context present")
    other_kw = [
        "progress notes",
        "nursing notes",
        "observation chart",
        "vital signs",
        "medication chart",
        "pathology",
        "blood test",
        "laboratory",
        "ambulance",
        "paramedic",
    ]
    hits = _contains_any(text, other_kw)
    if hits:
        cs.score += 0.3
        cs.reasons.append(f"hospital-other keywords: {hits}")
    return cs


def _score_standalone_radiology(text: str) -> _ClassificationScore:
    cs = _ClassificationScore(DocumentType.RADIOLOGY, 0.0)
    rad_kw = [
        "radiology report",
        "imaging report",
        "x-ray",
        "xray",
        "ct scan",
        "mri",
        "ultrasound",
        "radiograph",
        "fluoroscopy",
        "nuclear medicine",
        "pet scan",
        "bone scan",
        "emg",
        "nerve conduction",
        "ncv",
        "electromyography",
    ]
    hits = _contains_any(text, rad_kw)
    if hits:
        cs.score += 0.5
        cs.reasons.append(f"radiology keywords: {hits}")
    findings_kw = [
        "impression",
        "findings",
        "clinical indication",
        "comparison",
        "technique",
        "radiologist",
        "reported by",
    ]
    hits = _contains_any(text, findings_kw)
    if hits:
        cs.score += 0.25
        cs.reasons.append(f"findings structure: {hits}")
    # Penalize if hospital context is strong (more likely ED radiology)
    if _has_hospital_context(text):
        cs.score -= 0.15
        cs.reasons.append("hospital context penalty (may be ED radiology)")
    # Penalize if this looks like clinical notes referencing radiology
    notes_context = [
        "clinical notes",
        "presenting complaint",
        "refer for",
        "plan:",
        "assessment:",
    ]
    if _contains_any(text, notes_context):
        cs.score -= 0.3
        cs.reasons.append("clinical notes context penalty")
    return cs


def _score_standalone_operation(text: str) -> _ClassificationScore:
    cs = _ClassificationScore(DocumentType.OPERATION_REPORT, 0.0)
    op_kw = [
        "operation report",
        "operative report",
        "surgical report",
        "procedure report",
        "surgeon",
        "operative findings",
        "operative procedure",
    ]
    hits = _contains_any(text, op_kw)
    if hits:
        cs.score += 0.5
        cs.reasons.append(f"operation keywords: {hits}")
    detail_kw = [
        "incision",
        "excision",
        "suture",
        "specimen",
        "anaesthetist",
        "anesthetist",
        "post-operative",
        "postoperative",
    ]
    hits = _contains_any(text, detail_kw)
    if hits:
        cs.score += 0.25
        cs.reasons.append(f"surgical detail: {hits}")
    if _has_hospital_context(text):
        cs.score -= 0.15
        cs.reasons.append("hospital context penalty (may be ED operation)")
    return cs


def _score_treater_letter_non_allied(text: str) -> _ClassificationScore:
    cs = _ClassificationScore(DocumentType.TREATER_LETTER_NON_ALLIED, 0.0)
    letter_kw = [
        "dear ",
        "to whom it may concern",
        "re:",
        "regarding:",
        "thank you for referring",
        "i reviewed",
        "i saw",
        "i have seen",
        "this patient",
        "this gentleman",
        "this lady",
        "yours sincerely",
        "yours faithfully",
        "kind regards",
    ]
    hits = _contains_any(text, letter_kw)
    if hits:
        cs.score += 0.3
        cs.reasons.append(f"letter format keywords: {hits}")
    # Strong letter structure: salutation + closing = very likely a letter
    salutation = _contains_any(text, ["dear "])
    closing = _contains_any(
        text, ["yours sincerely", "yours faithfully", "kind regards"]
    )
    if salutation and closing:
        cs.score += 0.15
        cs.reasons.append("strong letter structure (salutation + closing)")
    medical_kw = [
        "dr ",
        "doctor",
        "physician",
        "general practitioner",
        "gp ",
        "specialist",
        "surgeon",
        "consultant",
        "registrar",
        "professor",
        "prof ",
    ]
    hits = _contains_any(text, medical_kw)
    if hits:
        cs.score += 0.2
        cs.reasons.append(f"medical practitioner indicators: {hits}")
    if _is_allied_health(text):
        cs.score -= 0.4
        cs.reasons.append("allied health detected - penalizing non-allied")
    if _has_legal_addressee(text) and _has_non_treating_practitioner_indicators(text):
        cs.score -= 0.4
        cs.reasons.append("medicolegal indicators - penalizing treater letter")
    return cs


def _score_treater_letter_allied(text: str) -> _ClassificationScore:
    cs = _ClassificationScore(DocumentType.TREATER_LETTER_ALLIED, 0.0)
    letter_kw = [
        "dear ",
        "to whom it may concern",
        "re:",
        "regarding:",
        "thank you for referring",
        "yours sincerely",
        "yours faithfully",
        "kind regards",
    ]
    hits = _contains_any(text, letter_kw)
    if hits:
        cs.score += 0.3
        cs.reasons.append(f"letter format keywords: {hits}")
    if _is_allied_health(text):
        cs.score += 0.35
        cs.reasons.append("allied health profession detected")
    else:
        cs.score -= 0.3
        cs.reasons.append("no allied health indicators")
    if _has_legal_addressee(text) and _has_non_treating_practitioner_indicators(text):
        cs.score -= 0.4
        cs.reasons.append("medicolegal indicators - penalizing treater letter")
    return cs


def _score_referral_letter(text: str) -> _ClassificationScore:
    cs = _ClassificationScore(DocumentType.REFERRAL_LETTER, 0.0)
    ref_kw = [
        "referral",
        "refer to",
        "referring to",
        "please see",
        "please review",
        "i am referring",
        "i would be grateful",
        "would appreciate your",
        "kindly see",
        "requesting assessment",
        "request for consultation",
    ]
    hits = _contains_any(text, ref_kw)
    if hits:
        cs.score += 0.45
        cs.reasons.append(f"referral keywords: {hits}")
    letter_kw = ["dear ", "yours sincerely", "yours faithfully", "kind regards"]
    hits = _contains_any(text, letter_kw)
    if hits:
        cs.score += 0.2
        cs.reasons.append(f"letter format: {hits}")
    return cs


def _score_clinical_notes_non_allied(text: str) -> _ClassificationScore:
    cs = _ClassificationScore(DocumentType.CLINICAL_NOTES_NON_ALLIED, 0.0)
    notes_kw = [
        "clinical notes",
        "consultation notes",
        "progress notes",
        "patient notes",
        "medical record",
        "medical notes",
        "history of presenting",
        "presenting complaint",
        "examination",
        "assessment",
        "plan",
        "soap",
        "subjective",
        "objective",
    ]
    hits = _contains_any(text, notes_kw)
    if hits:
        cs.score += 0.35
        cs.reasons.append(f"clinical notes keywords: {hits}")
    medical_kw = ["dr ", "doctor", "gp ", "general practitioner"]
    hits = _contains_any(text, medical_kw)
    if hits:
        cs.score += 0.15
        cs.reasons.append(f"medical practitioner: {hits}")
    # GP-specific boost
    gp_indicators = ["general practitioner", "gp ", "family doctor", "family medicine"]
    gp_hits = _contains_any(text, gp_indicators)
    if gp_hits:
        cs.score += 0.15
        cs.reasons.append(f"GP-specific indicators: {gp_hits}")
    if _is_allied_health(text):
        cs.score -= 0.35
        cs.reasons.append("allied health detected - penalizing non-allied notes")
    # Penalize letter format (these are notes, not letters)
    letter_kw = ["dear ", "yours sincerely", "yours faithfully"]
    if _contains_any(text, letter_kw):
        cs.score -= 0.2
        cs.reasons.append("letter format detected - less likely to be notes")
    return cs


def _score_clinical_notes_allied(text: str) -> _ClassificationScore:
    cs = _ClassificationScore(DocumentType.CLINICAL_NOTES_ALLIED, 0.0)
    notes_kw = [
        "clinical notes",
        "treatment notes",
        "progress notes",
        "session notes",
        "assessment",
        "treatment plan",
        "treatment record",
        "initial assessment",
    ]
    hits = _contains_any(text, notes_kw)
    if hits:
        cs.score += 0.3
        cs.reasons.append(f"clinical notes keywords: {hits}")
    if _is_allied_health(text):
        cs.score += 0.35
        cs.reasons.append("allied health profession detected")
    else:
        cs.score -= 0.3
        cs.reasons.append("no allied health indicators")
    letter_kw = ["dear ", "yours sincerely", "yours faithfully"]
    if _contains_any(text, letter_kw):
        cs.score -= 0.2
        cs.reasons.append("letter format detected - less likely to be notes")
    return cs


def _score_specialist_clinical_notes(text: str) -> _ClassificationScore:
    cs = _ClassificationScore(DocumentType.SPECIALIST_CLINICAL_NOTES, 0.0)
    spec_kw = [
        "specialist",
        "consultant",
        "registrar",
        "orthopaedic",
        "orthopedic",
        "neurologist",
        "neurology",
        "cardiologist",
        "cardiology",
        "rheumatologist",
        "rheumatology",
        "psychiatrist",
        "psychiatry",
        "endocrinologist",
        "gastroenterologist",
        "dermatologist",
        "urologist",
        "nephrologist",
        "oncologist",
        "haematologist",
        "hematologist",
        "immunologist",
        "ophthalmologist",
        "ent ",
        "ear nose throat",
        "pain specialist",
        "pain management",
        "rehabilitation",
        "professor",
        "prof ",
    ]
    hits = _contains_any(text, spec_kw)
    if hits:
        cs.score += 0.35
        cs.reasons.append(f"specialist indicators: {hits}")
    notes_kw = [
        "clinical notes",
        "consultation notes",
        "assessment",
        "examination",
        "impression",
        "plan",
    ]
    hits = _contains_any(text, notes_kw)
    if hits:
        cs.score += 0.2
        cs.reasons.append(f"clinical notes structure: {hits}")
    if _is_allied_health(text):
        cs.score -= 0.3
        cs.reasons.append("allied health detected - penalizing specialist")
    # Penalize letter format (specialist clinical notes are notes, not letters)
    letter_kw = ["dear ", "yours sincerely", "yours faithfully", "kind regards"]
    if _contains_any(text, letter_kw):
        cs.score -= 0.2
        cs.reasons.append("letter format detected - less likely to be notes")
    return cs


def _score_medicolegal_report(text: str) -> _ClassificationScore:
    cs = _ClassificationScore(DocumentType.MEDICOLEGAL_REPORT, 0.0)

    # Both criteria must be met for medicolegal classification:
    # 1. Non-treating practitioner
    # 2. Addressed to legal/insurance entity
    has_non_treating = _has_non_treating_practitioner_indicators(text)
    has_legal = _has_legal_addressee(text)

    if has_non_treating and has_legal:
        cs.score += 0.7
        cs.reasons.append(
            "BOTH medicolegal criteria met: non-treating practitioner "
            "AND legal/insurance addressee"
        )
    elif has_non_treating:
        cs.score += 0.3
        cs.reasons.append(
            "non-treating practitioner indicators found, "
            "but no clear legal addressee"
        )
    elif has_legal:
        cs.score += 0.15
        cs.reasons.append(
            "legal addressee found, but no clear non-treating "
            "practitioner indicators"
        )

    ml_kw = [
        "medicolegal",
        "medico-legal",
        "independent medical",
        "permanent impairment",
        "whole person impairment",
        "ama guides",
        "impairment rating",
        "fit for work",
        "fitness for duty",
        "capacity assessment",
    ]
    hits = _contains_any(text, ml_kw)
    if hits:
        cs.score += 0.2
        cs.reasons.append(f"medicolegal terminology: {hits}")
    return cs


def _score_medical_panel_report(text: str) -> _ClassificationScore:
    cs = _ClassificationScore(DocumentType.MEDICAL_PANEL_REPORT, 0.0)

    panel_kw = [
        "medical panel",
        "panel report",
        "panel opinion",
        "panel members",
        "panel examination",
        "panel assessment",
        "medical panels victoria",
    ]
    hits = _contains_any(text, panel_kw)
    if hits:
        cs.score += 0.6
        cs.reasons.append(f"medical panel keywords: {hits}")

    has_non_treating = _has_non_treating_practitioner_indicators(text)
    has_legal = _has_legal_addressee(text)
    if has_non_treating and has_legal:
        cs.score += 0.25
        cs.reasons.append("medicolegal criteria also met")

    return cs


def _score_affidavit(text: str) -> _ClassificationScore:
    cs = _ClassificationScore(DocumentType.AFFIDAVIT, 0.0)
    aff_kw = [
        "affidavit",
        "sworn",
        "deponent",
        "solemnly declare",
        "affirm",
        "oath",
        "statutory declaration",
        "before me",
        "witness my hand",
        "notary",
        "commissioner for taking affidavits",
        "justice of the peace",
    ]
    hits = _contains_any(text, aff_kw)
    if hits:
        cs.score += 0.7
        cs.reasons.append(f"affidavit keywords: {hits}")
    legal_structure = [
        "in the matter of",
        "between:",
        "plaintiff",
        "defendant",
        "applicant",
        "respondent",
        "county court",
        "supreme court",
        "magistrates",
    ]
    hits = _contains_any(text, legal_structure)
    if hits:
        cs.score += 0.2
        cs.reasons.append(f"legal document structure: {hits}")
    return cs


# All scoring functions in evaluation order
_SCORING_FUNCTIONS = [
    _score_affidavit,
    _score_medical_panel_report,
    _score_medicolegal_report,
    _score_ed_discharge,
    _score_ed_operation,
    _score_ed_radiology,
    _score_ed_outpatient,
    _score_ed_other,
    _score_standalone_radiology,
    _score_standalone_operation,
    _score_referral_letter,
    _score_treater_letter_non_allied,
    _score_treater_letter_allied,
    _score_clinical_notes_non_allied,
    _score_clinical_notes_allied,
    _score_specialist_clinical_notes,
]

# Minimum score to accept a classification (below this -> "Other")
_MIN_CONFIDENCE_THRESHOLD = 0.25


def classify_page(page: PageContent) -> tuple[DocumentType, float, str]:
    """Classify a single page and return (type, confidence, reasoning).

    Args:
        page: Extracted page content.

    Returns:
        Tuple of (DocumentType, confidence_score, reasoning_text).
    """
    if not page.text.strip():
        return (
            DocumentType.OTHER,
            0.0,
            "Page has no extractable text content",
        )

    text = _normalize(page.text)

    scores: list[_ClassificationScore] = []
    for fn in _SCORING_FUNCTIONS:
        result = fn(text)
        if result.score > 0:
            scores.append(result)

    if not scores:
        return (
            DocumentType.OTHER,
            0.0,
            "No classification patterns matched",
        )

    # Sort by score descending
    scores.sort(key=lambda s: s.score, reverse=True)
    best = scores[0]

    if best.score < _MIN_CONFIDENCE_THRESHOLD:
        reasoning = (
            f"Best match was {best.document_type.value} "
            f"(score={best.score:.3f}) but below threshold "
            f"({_MIN_CONFIDENCE_THRESHOLD}). Reasons: {'; '.join(best.reasons)}"
        )
        return (DocumentType.OTHER, best.score, reasoning)

    # Normalize confidence to 0-1 range (cap at 1.0)
    confidence = min(best.score, 1.0)
    reasoning = f"Classified as {best.document_type.value}. " + "; ".join(
        best.reasons
    )

    logger.debug(
        "Page %d -> %s (%.3f): %s",
        page.page_number,
        best.document_type.value,
        confidence,
        reasoning,
    )

    return (best.document_type, confidence, reasoning)
