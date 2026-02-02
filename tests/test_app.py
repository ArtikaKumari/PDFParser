"""Tests for the Flask web application and API endpoints."""

import io
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from pdf_classifier.app import create_app
from pdf_classifier.models import DocumentResult, PageClassification
from pdf_classifier.taxonomy import DocumentType, PriorityLevel, SourceType


@pytest.fixture
def app(tmp_path):
    """Create a test Flask application."""
    app = create_app(upload_dir=tmp_path / "uploads")
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()


def _make_fake_result(filename: str = "test.pdf") -> DocumentResult:
    result = DocumentResult(file_path=filename, total_pages=2)
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
    ]
    return result


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"


class TestIndexPage:
    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"PDF Document Classifier" in resp.data
        assert b"text/html" in resp.content_type.encode()


class TestClassifyEndpoint:
    def test_no_file_returns_400(self, client):
        resp = client.post("/api/classify")
        assert resp.status_code == 400
        data = resp.get_json()
        assert "No file provided" in data["error"]

    def test_empty_filename_returns_400(self, client):
        resp = client.post(
            "/api/classify",
            data={"file": (io.BytesIO(b""), "")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "No file selected" in data["error"]

    def test_non_pdf_returns_400(self, client):
        resp = client.post(
            "/api/classify",
            data={"file": (io.BytesIO(b"hello"), "test.txt")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "PDF" in data["error"]

    @patch("pdf_classifier.app.classify_document")
    @patch("pdf_classifier.app.route_batch")
    def test_successful_classification(
        self, mock_route, mock_classify, client
    ):
        fake_result = _make_fake_result()
        mock_classify.return_value = fake_result
        mock_route.return_value = []

        pdf_bytes = b"%PDF-1.4 fake content"
        resp = client.post(
            "/api/classify",
            data={"file": (io.BytesIO(pdf_bytes), "document.pdf")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total_pages"] == 2
        assert data["original_filename"] == "document.pdf"
        assert len(data["pages"]) == 2
        assert "classification_summary" in data
        assert "source_type_summary" in data
        assert "priority_summary" in data
        assert "report_text" in data

    @patch("pdf_classifier.app.classify_document")
    @patch("pdf_classifier.app.route_batch")
    def test_classification_returns_processing_results(
        self, mock_route, mock_classify, client
    ):
        fake_result = _make_fake_result()
        mock_classify.return_value = fake_result

        from pdf_classifier.processor import ProcessingResult

        mock_route.return_value = [
            ProcessingResult(
                page_number=1,
                processor_name="ed_discharge_processor",
                success=True,
            ),
            ProcessingResult(
                page_number=2,
                processor_name="standalone_radiology_processor",
                success=True,
            ),
        ]

        pdf_bytes = b"%PDF-1.4 fake content"
        resp = client.post(
            "/api/classify",
            data={"file": (io.BytesIO(pdf_bytes), "document.pdf")},
            content_type="multipart/form-data",
        )
        data = resp.get_json()
        assert "processing_results" in data
        assert len(data["processing_results"]) == 2
        assert data["processing_results"][0]["processor_name"] == "ed_discharge_processor"

    @patch("pdf_classifier.app.classify_document")
    def test_classification_error_returns_500(self, mock_classify, client):
        mock_classify.side_effect = RuntimeError("Parse failed")

        pdf_bytes = b"%PDF-1.4 fake content"
        resp = client.post(
            "/api/classify",
            data={"file": (io.BytesIO(pdf_bytes), "bad.pdf")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 500
        data = resp.get_json()
        assert "Classification failed" in data["error"]

    @patch("pdf_classifier.app.classify_document")
    @patch("pdf_classifier.app.route_batch")
    def test_uploaded_file_is_cleaned_up(
        self, mock_route, mock_classify, client, app
    ):
        fake_result = _make_fake_result()
        mock_classify.return_value = fake_result
        mock_route.return_value = []

        pdf_bytes = b"%PDF-1.4 fake content"
        client.post(
            "/api/classify",
            data={"file": (io.BytesIO(pdf_bytes), "document.pdf")},
            content_type="multipart/form-data",
        )

        upload_dir = Path(app.config["UPLOAD_DIR"])
        pdf_files = list(upload_dir.glob("*.pdf"))
        assert len(pdf_files) == 0, "Upload should be deleted after processing"


class TestClassifyOnlyEndpoint:
    def test_no_file_returns_400(self, client):
        resp = client.post("/api/classify-only")
        assert resp.status_code == 400

    def test_non_pdf_returns_400(self, client):
        resp = client.post(
            "/api/classify-only",
            data={"file": (io.BytesIO(b"hello"), "test.docx")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    @patch("pdf_classifier.app.classify_document")
    def test_successful_classify_only(self, mock_classify, client):
        fake_result = _make_fake_result()
        mock_classify.return_value = fake_result

        pdf_bytes = b"%PDF-1.4 fake content"
        resp = client.post(
            "/api/classify-only",
            data={"file": (io.BytesIO(pdf_bytes), "document.pdf")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total_pages"] == 2
        assert "processing_results" not in data
        assert "report_text" in data

    @patch("pdf_classifier.app.classify_document")
    def test_classify_only_error_returns_500(self, mock_classify, client):
        mock_classify.side_effect = ValueError("Bad PDF")

        pdf_bytes = b"%PDF-1.4 bad"
        resp = client.post(
            "/api/classify-only",
            data={"file": (io.BytesIO(pdf_bytes), "bad.pdf")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 500

    @patch("pdf_classifier.app.classify_document")
    def test_classify_only_cleans_up_file(self, mock_classify, client, app):
        fake_result = _make_fake_result()
        mock_classify.return_value = fake_result

        pdf_bytes = b"%PDF-1.4 content"
        client.post(
            "/api/classify-only",
            data={"file": (io.BytesIO(pdf_bytes), "test.pdf")},
            content_type="multipart/form-data",
        )

        upload_dir = Path(app.config["UPLOAD_DIR"])
        assert len(list(upload_dir.glob("*.pdf"))) == 0


class TestResponseStructure:
    @patch("pdf_classifier.app.classify_document")
    @patch("pdf_classifier.app.route_batch")
    def test_page_data_structure(self, mock_route, mock_classify, client):
        fake_result = _make_fake_result()
        mock_classify.return_value = fake_result
        mock_route.return_value = []

        pdf_bytes = b"%PDF-1.4 content"
        resp = client.post(
            "/api/classify",
            data={"file": (io.BytesIO(pdf_bytes), "doc.pdf")},
            content_type="multipart/form-data",
        )
        data = resp.get_json()
        page = data["pages"][0]

        assert "page_number" in page
        assert "document_type" in page
        assert "category" in page
        assert "source_type" in page
        assert "priority_level" in page
        assert "confidence" in page
        assert "flags" in page
        assert "needs_manual_review" in page
        assert "classification_reasoning" in page

    @patch("pdf_classifier.app.classify_document")
    @patch("pdf_classifier.app.route_batch")
    def test_response_is_valid_json(self, mock_route, mock_classify, client):
        fake_result = _make_fake_result()
        mock_classify.return_value = fake_result
        mock_route.return_value = []

        pdf_bytes = b"%PDF-1.4 content"
        resp = client.post(
            "/api/classify",
            data={"file": (io.BytesIO(pdf_bytes), "doc.pdf")},
            content_type="multipart/form-data",
        )
        # Should not raise
        data = json.loads(resp.data)
        assert isinstance(data, dict)
