"""Flask web application for PDF document classification.

Provides a REST API and a lightweight web UI for uploading PDFs and
viewing classification results.
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, render_template, send_from_directory

from pdf_classifier.main import classify_document, process_document
from pdf_classifier.output import format_report
from pdf_classifier.processor import route_batch

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "pdf_classifier_uploads"
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB


def create_app(upload_dir: Path | None = None) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    )
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    upload_path = upload_dir or UPLOAD_DIR
    upload_path.mkdir(parents=True, exist_ok=True)
    app.config["UPLOAD_DIR"] = str(upload_path)

    _register_routes(app)
    return app


def _register_routes(app: Flask) -> None:
    """Register all API and UI routes."""

    @app.route("/")
    def index():
        """Serve the web UI."""
        return render_template("index.html")

    @app.route("/api/health")
    def health():
        """Health check endpoint."""
        return jsonify({"status": "ok"})

    @app.route("/api/classify", methods=["POST"])
    def classify():
        """Upload a PDF and classify its pages.

        Accepts multipart/form-data with a 'file' field containing the PDF.
        Returns JSON classification results.
        """
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "File must be a PDF"}), 400

        upload_dir = Path(app.config["UPLOAD_DIR"])
        file_id = uuid.uuid4().hex[:12]
        safe_name = f"{file_id}.pdf"
        file_path = upload_dir / safe_name

        try:
            file.save(str(file_path))
            logger.info("Saved upload to %s", file_path)

            result = classify_document(file_path)
            processing_results = route_batch(result.page_classifications)
            report_text = format_report(result, processing_results)

            response = result.to_dict()
            response["report_text"] = report_text
            response["original_filename"] = file.filename
            response["processing_results"] = [
                {
                    "page_number": pr.page_number,
                    "processor_name": pr.processor_name,
                    "success": pr.success,
                    "error_message": pr.error_message,
                }
                for pr in processing_results
            ]

            return jsonify(response)

        except Exception as exc:
            logger.exception("Classification failed for %s", file.filename)
            return jsonify({"error": f"Classification failed: {exc}"}), 500

        finally:
            if file_path.exists():
                file_path.unlink()

    @app.route("/api/classify-only", methods=["POST"])
    def classify_only():
        """Upload a PDF and classify without downstream processing.

        Same as /api/classify but skips the processing router.
        """
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "File must be a PDF"}), 400

        upload_dir = Path(app.config["UPLOAD_DIR"])
        file_id = uuid.uuid4().hex[:12]
        safe_name = f"{file_id}.pdf"
        file_path = upload_dir / safe_name

        try:
            file.save(str(file_path))
            result = classify_document(file_path)
            report_text = format_report(result)

            response = result.to_dict()
            response["report_text"] = report_text
            response["original_filename"] = file.filename

            return jsonify(response)

        except Exception as exc:
            logger.exception("Classification failed for %s", file.filename)
            return jsonify({"error": f"Classification failed: {exc}"}), 500

        finally:
            if file_path.exists():
                file_path.unlink()


def run() -> None:
    """Entry point for the web server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    logger.info("Starting PDF Classifier web server on %s:%d", host, port)
    app.run(host=host, port=port, debug=os.environ.get("FLASK_DEBUG") == "1")


if __name__ == "__main__":
    run()
