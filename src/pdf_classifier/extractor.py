"""PDF text extraction with OCR fallback support."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)

# Minimum character count to consider a page as having usable text
_MIN_TEXT_LENGTH = 30


@dataclass
class PageContent:
    """Extracted content from a single PDF page."""

    page_number: int
    text: str
    extraction_method: str  # "text" or "ocr"
    has_images: bool
    word_count: int


def _try_ocr(pdf_path: str, page_number: int) -> str:
    """Attempt OCR extraction for a page. Returns empty string on failure."""
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        logger.warning(
            "OCR dependencies not available (pdf2image/pytesseract). "
            "Install them for OCR support."
        )
        return ""

    try:
        images = convert_from_path(
            pdf_path,
            first_page=page_number,
            last_page=page_number,
            dpi=300,
        )
        if images:
            text: str = pytesseract.image_to_string(images[0])
            return text.strip()
    except Exception:
        logger.exception("OCR failed for page %d of %s", page_number, pdf_path)
    return ""


def extract_page(
    pdf_path: str, page: pdfplumber.page.Page, page_number: int
) -> PageContent:
    """Extract text content from a single PDF page.

    Tries direct text extraction first, falls back to OCR if the page
    appears to be a scanned image with insufficient embedded text.
    """
    raw_text = page.extract_text() or ""
    text = raw_text.strip()
    has_images = bool(page.images)

    if len(text) >= _MIN_TEXT_LENGTH:
        words = text.split()
        return PageContent(
            page_number=page_number,
            text=text,
            extraction_method="text",
            has_images=has_images,
            word_count=len(words),
        )

    # Text extraction yielded little content - try OCR if images are present
    if has_images:
        logger.info(
            "Page %d has insufficient text (%d chars), attempting OCR",
            page_number,
            len(text),
        )
        ocr_text = _try_ocr(pdf_path, page_number)
        if len(ocr_text) >= _MIN_TEXT_LENGTH:
            words = ocr_text.split()
            return PageContent(
                page_number=page_number,
                text=ocr_text,
                extraction_method="ocr",
                has_images=has_images,
                word_count=len(words),
            )

    # Return whatever text we have, even if minimal
    words = text.split() if text else []
    method = "text" if text else "empty"
    return PageContent(
        page_number=page_number,
        text=text,
        extraction_method=method,
        has_images=has_images,
        word_count=len(words),
    )


def extract_all_pages(pdf_path: str | Path) -> list[PageContent]:
    """Extract text content from all pages of a PDF document.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of PageContent objects, one per page.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the file is not a valid PDF.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {path}")

    pages: list[PageContent] = []
    try:
        with pdfplumber.open(path) as pdf:
            logger.info(
                "Opened PDF: %s (%d pages)", path.name, len(pdf.pages)
            )
            for i, page in enumerate(pdf.pages):
                page_number = i + 1  # 1-based page numbering
                content = extract_page(str(path), page, page_number)
                pages.append(content)
                logger.debug(
                    "Page %d: %d words via %s",
                    page_number,
                    content.word_count,
                    content.extraction_method,
                )
    except Exception as exc:
        if "PDF" in str(type(exc).__name__) or "pdf" in str(exc).lower():
            raise ValueError(f"Failed to parse PDF: {exc}") from exc
        raise

    return pages
