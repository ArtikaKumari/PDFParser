"""Tests for the page classification engine."""

import pytest

from pdf_classifier.classifier import classify_page
from pdf_classifier.extractor import PageContent
from pdf_classifier.taxonomy import DocumentType


def _make_page(text: str, page_number: int = 1) -> PageContent:
    """Helper to create a PageContent for testing."""
    return PageContent(
        page_number=page_number,
        text=text,
        extraction_method="text",
        has_images=False,
        word_count=len(text.split()),
    )


class TestEmptyPages:
    def test_empty_text_returns_other(self):
        page = _make_page("")
        doc_type, confidence, _ = classify_page(page)
        assert doc_type == DocumentType.OTHER
        assert confidence == 0.0

    def test_whitespace_only_returns_other(self):
        page = _make_page("   \n\n\t  ")
        doc_type, confidence, _ = classify_page(page)
        assert doc_type == DocumentType.OTHER


class TestDischargeSummaryClassification:
    def test_basic_discharge_summary(self):
        text = """
        DISCHARGE SUMMARY
        Hospital: Royal Melbourne Hospital
        Ward: 4B
        Admission Date: 01/03/2024
        Discharge Date: 05/03/2024
        Principal Diagnosis: Left tibial plateau fracture
        Procedures Performed: ORIF left tibia
        Discharge Medications: Paracetamol 1g QID, Endone 5mg PRN
        Follow up: Orthopaedic clinic in 6 weeks
        """
        page = _make_page(text)
        doc_type, confidence, _ = classify_page(page)
        assert doc_type == DocumentType.ED_DISCHARGE_SUMMARY
        assert confidence > 0.5

    def test_discharge_letter(self):
        text = """
        Discharge Letter
        Emergency Department
        Patient was admitted with chest pain.
        Discharged home with follow-up appointment.
        Condition on discharge: stable.
        """
        page = _make_page(text)
        doc_type, _, _ = classify_page(page)
        assert doc_type == DocumentType.ED_DISCHARGE_SUMMARY


class TestOperationReportClassification:
    def test_hospital_operation_report(self):
        text = """
        OPERATION REPORT
        Hospital: St Vincent's Hospital
        Surgeon: Dr A. Smith
        Anaesthetist: Dr B. Jones
        Procedure: Arthroscopic ACL reconstruction
        Operative Findings: Complete ACL tear
        Incision: Standard anteromedial portal
        Post-operative instructions: NWB 6 weeks
        """
        page = _make_page(text)
        doc_type, confidence, _ = classify_page(page)
        assert doc_type in (
            DocumentType.ED_OPERATION_REPORT,
            DocumentType.OPERATION_REPORT,
        )
        assert confidence > 0.4

    def test_standalone_operation_report(self):
        text = """
        OPERATIVE REPORT
        Surgeon: Dr C. Williams
        Procedure: Right carpal tunnel release
        Operative findings: Thickened transverse carpal ligament
        Incision: 3cm longitudinal incision over thenar crease
        Specimen: None
        Post-operative plan: Review in 2 weeks
        """
        page = _make_page(text)
        doc_type, confidence, _ = classify_page(page)
        assert doc_type in (
            DocumentType.ED_OPERATION_REPORT,
            DocumentType.OPERATION_REPORT,
        )
        assert confidence > 0.4


class TestRadiologyClassification:
    def test_hospital_radiology(self):
        text = """
        RADIOLOGY REPORT
        Hospital: Alfred Hospital
        Modality: CT scan
        Clinical Indication: Trauma - fall from height
        Findings: Comminuted fracture of the left calcaneus.
        Impression: Left calcaneal fracture, surgical consultation recommended.
        Radiologist: Dr D. Lee
        """
        page = _make_page(text)
        doc_type, confidence, _ = classify_page(page)
        assert doc_type in (
            DocumentType.ED_RADIOLOGY,
            DocumentType.RADIOLOGY,
        )
        assert confidence > 0.4

    def test_standalone_mri_report(self):
        text = """
        MRI Report
        Referred by: Dr Smith
        Clinical Indication: Low back pain, ?disc pathology
        Technique: Sagittal and axial T1 and T2-weighted sequences
        Findings: L4/5 disc protrusion with left foraminal extension
        Impression: L4/5 disc protrusion causing left L5 nerve root compression
        Reported by: Dr E. Chen, Radiologist
        """
        page = _make_page(text)
        doc_type, confidence, _ = classify_page(page)
        assert doc_type == DocumentType.RADIOLOGY
        assert confidence > 0.4

    def test_emg_ncv_study(self):
        text = """
        EMG / Nerve Conduction Study Report
        Referred by: Dr Smith
        Clinical Indication: Left upper limb weakness
        Nerve conduction studies show reduced amplitude in the left median nerve.
        EMG findings: Chronic denervation changes in C5-C6 myotomes.
        Impression: Left C5-C6 radiculopathy.
        """
        page = _make_page(text)
        doc_type, confidence, _ = classify_page(page)
        assert doc_type == DocumentType.RADIOLOGY
        assert confidence > 0.3


class TestOutpatientNotesClassification:
    def test_outpatient_clinic_review(self):
        text = """
        OUTPATIENT CLINIC REVIEW
        Hospital: Epworth Hospital
        Consultant: Dr F. Brown
        Clinic: Orthopaedic Outpatient Clinic
        Date: 15/04/2024
        Review appointment following left knee ORIF.
        Examination: Good range of motion. Wound healed.
        Plan: Continue physiotherapy. Review in 3 months.
        """
        page = _make_page(text)
        doc_type, confidence, _ = classify_page(page)
        assert doc_type == DocumentType.ED_OUTPATIENT_NOTES
        assert confidence > 0.3


class TestCorrespondenceClassification:
    def test_treater_letter_non_allied(self):
        text = """
        Dr G. Harrison
        General Practitioner
        123 Collins Street, Melbourne VIC 3000

        Dear Dr Smith,

        Re: John Patient DOB 01/01/1980

        Thank you for referring this patient for review.
        I reviewed Mr Patient on 20/04/2024.
        Examination revealed limited lumbar flexion.
        Yours sincerely,
        Dr G. Harrison
        """
        page = _make_page(text)
        doc_type, confidence, _ = classify_page(page)
        assert doc_type == DocumentType.TREATER_LETTER_NON_ALLIED
        assert confidence > 0.3

    def test_treater_letter_allied_health(self):
        text = """
        Jane Brown
        Physiotherapist
        ABC Physiotherapy Clinic

        Dear Dr Smith,

        Re: John Patient

        I have been treating Mr Patient for his low back pain.
        This patient presented with reduced lumbar range of motion.
        Yours sincerely,
        Jane Brown
        Physiotherapist
        """
        page = _make_page(text)
        doc_type, confidence, _ = classify_page(page)
        assert doc_type == DocumentType.TREATER_LETTER_ALLIED
        assert confidence > 0.3

    def test_referral_letter(self):
        text = """
        Dear Dr Williams,

        I am referring Mr Patient for your orthopaedic opinion.
        Please review this patient who has ongoing knee pain
        following a motor vehicle accident.
        I would be grateful for your assessment and management advice.

        Kind regards,
        Dr H. Taylor
        General Practitioner
        """
        page = _make_page(text)
        doc_type, confidence, _ = classify_page(page)
        assert doc_type == DocumentType.REFERRAL_LETTER
        assert confidence > 0.3


class TestClinicalNotesClassification:
    def test_gp_clinical_notes(self):
        text = """
        CLINICAL NOTES
        Dr I. Martinez, General Practitioner
        Date: 25/04/2024
        Presenting complaint: Ongoing low back pain
        History of presenting illness: Patient reports worsening pain
        Examination: Tenderness over L4/L5 region
        Assessment: Lumbar disc disease
        Plan: Refer for MRI, continue NSAIDs
        """
        page = _make_page(text)
        doc_type, confidence, _ = classify_page(page)
        assert doc_type == DocumentType.CLINICAL_NOTES_NON_ALLIED
        assert confidence > 0.3

    def test_allied_health_clinical_notes(self):
        text = """
        TREATMENT NOTES
        Physiotherapy Session Notes
        Therapist: Sarah Johnson, Physiotherapist
        Date: 28/04/2024
        Initial assessment of lumbar spine
        Treatment plan: Manual therapy, core strengthening exercises
        Session notes: Applied mobilisation to L4/L5 segment
        """
        page = _make_page(text)
        doc_type, confidence, _ = classify_page(page)
        assert doc_type == DocumentType.CLINICAL_NOTES_ALLIED
        assert confidence > 0.3

    def test_specialist_clinical_notes(self):
        text = """
        Consultation Notes
        Dr J. Orthopaedic Surgeon
        Specialist Consultation
        Orthopaedic assessment of left knee
        Examination: Positive Lachman test, positive pivot shift
        Impression: ACL rupture
        Plan: Arthroscopic ACL reconstruction
        """
        page = _make_page(text)
        doc_type, confidence, _ = classify_page(page)
        assert doc_type == DocumentType.SPECIALIST_CLINICAL_NOTES
        assert confidence > 0.3


class TestMedicolegalClassification:
    """Tests for the critical medicolegal vs treatment distinction."""

    def test_medicolegal_report_both_criteria(self):
        """Both criteria met: non-treating + legal addressee."""
        text = """
        MEDICOLEGAL REPORT
        Independent Medical Examination

        Prepared at the request of: Slater & Gordon Lawyers
        Instructing solicitor: Ms A. Lawyer

        I was asked to examine Mr Patient on 01/05/2024.
        I am not involved in the ongoing treatment of this patient.

        Whole person impairment assessment: 10% WPI
        This is a medico-legal opinion prepared for the purpose of
        the TAC claim.
        """
        page = _make_page(text)
        doc_type, confidence, _ = classify_page(page)
        assert doc_type == DocumentType.MEDICOLEGAL_REPORT
        assert confidence > 0.5

    def test_medicolegal_with_workcover(self):
        """WorkCover addressee with independent examiner."""
        text = """
        Independent Medical Assessment

        Prepared for: WorkCover Authority
        I was requested to assess Mr Patient regarding his
        permanent impairment following a workplace injury.
        Independent assessment performed on 15/05/2024.
        Impairment rating: 8% whole person impairment as per
        AMA Guides.
        """
        page = _make_page(text)
        doc_type, confidence, _ = classify_page(page)
        assert doc_type == DocumentType.MEDICOLEGAL_REPORT
        assert confidence > 0.5

    def test_treating_doctor_letter_not_medicolegal(self):
        """A treating doctor's letter should NOT be medicolegal."""
        text = """
        Dear Dr Smith,

        Re: John Patient

        I reviewed this patient in my clinic on 10/05/2024.
        He continues to have ongoing low back pain.
        I have prescribed physiotherapy.

        Yours sincerely,
        Dr K. Treater
        General Practitioner
        """
        page = _make_page(text)
        doc_type, _, _ = classify_page(page)
        assert doc_type != DocumentType.MEDICOLEGAL_REPORT
        assert doc_type != DocumentType.MEDICAL_PANEL_REPORT

    def test_medical_panel_report(self):
        text = """
        MEDICAL PANEL REPORT

        Medical Panels Victoria
        Panel Members: Dr X, Dr Y, Dr Z

        Panel examination conducted on 20/05/2024.
        At the request of the TAC, we examined Mr Patient.

        Panel opinion: The injuries are consistent with the
        described accident mechanism.
        """
        page = _make_page(text)
        doc_type, confidence, _ = classify_page(page)
        assert doc_type == DocumentType.MEDICAL_PANEL_REPORT
        assert confidence > 0.5


class TestAffidavitClassification:
    def test_affidavit(self):
        text = """
        AFFIDAVIT

        In the County Court of Victoria
        Between: John Patient (Plaintiff)
        And: XYZ Insurance (Defendant)

        I, John Patient, of Melbourne, solemnly declare and affirm:

        1. I was involved in a motor vehicle accident on 01/01/2023.
        2. I suffered injuries to my neck and back.

        Sworn before me this 1st day of June 2024.
        Justice of the Peace
        """
        page = _make_page(text)
        doc_type, confidence, _ = classify_page(page)
        assert doc_type == DocumentType.AFFIDAVIT
        assert confidence > 0.5

    def test_statutory_declaration(self):
        text = """
        STATUTORY DECLARATION

        I, Jane Witness, do solemnly declare that:
        1. I witnessed the accident on Smith Street.
        2. The vehicle ran the red light.

        Sworn at Melbourne on 15/06/2024
        Before me: Commissioner for taking affidavits
        """
        page = _make_page(text)
        doc_type, confidence, _ = classify_page(page)
        assert doc_type == DocumentType.AFFIDAVIT
        assert confidence > 0.5


class TestEdgeAndFallbackCases:
    def test_ambiguous_text_returns_reasonable_result(self):
        text = "Patient seen today. Back pain. Review 2 weeks."
        page = _make_page(text)
        doc_type, confidence, _ = classify_page(page)
        # Should get some classification or OTHER
        assert isinstance(doc_type, DocumentType)

    def test_unrelated_text_returns_other(self):
        text = """
        Today's lunch specials:
        1. Fish and chips
        2. Caesar salad
        3. Chicken parma
        """
        page = _make_page(text)
        doc_type, _, _ = classify_page(page)
        assert doc_type == DocumentType.OTHER

    def test_confidence_is_between_0_and_1(self):
        text = """
        DISCHARGE SUMMARY
        Hospital admission for fracture management.
        Principal diagnosis: Left wrist fracture.
        Discharged with follow-up.
        """
        page = _make_page(text)
        _, confidence, _ = classify_page(page)
        assert 0.0 <= confidence <= 1.0

    def test_reasoning_is_non_empty_for_classified_pages(self):
        text = """
        Radiology Report
        MRI of lumbar spine
        Findings: Disc protrusion at L4/5
        Impression: L4/5 disc herniation
        """
        page = _make_page(text)
        _, _, reasoning = classify_page(page)
        assert len(reasoning) > 0
