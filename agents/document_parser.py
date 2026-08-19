import os
from parsers.pdf_parser import parse_pdf
from parsers.docx_parser import parse_docx
from parsers.pptx_parser import parse_pptx
from parsers.image_parser import parse_image
from utils.chunking import chunk_text
from models.schemas import ParsedDocument
from config import settings


def detect_file_type(filename: str) -> str | None:
    ext = os.path.splitext(filename)[1].lower()
    return settings.ALLOWED_EXTENSIONS.get(ext)


def parse_document(file_path: str, filename: str) -> tuple[ParsedDocument, list[dict]]:
    doc_type = detect_file_type(filename)
    if not doc_type:
        raise ValueError(f"Unsupported file type: {filename}")

    if doc_type == "pdf":
        parsed = parse_pdf(file_path)
    elif doc_type == "docx":
        parsed = parse_docx(file_path)
    elif doc_type == "pptx":
        parsed = parse_pptx(file_path)
    elif doc_type == "image":
        parsed = parse_image(file_path)
    else:
        raise ValueError(f"Unknown document type: {doc_type}")

    chunks = chunk_text(parsed.content)
    return parsed, chunks
