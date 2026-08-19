import pdf_inspector
from models.schemas import ParsedDocument


def parse_pdf(file_path: str) -> ParsedDocument:
    result = pdf_inspector.process_pdf(file_path)

    content = result.markdown or ""

    if not content.strip():
        content = f"[PDF has {result.page_count} pages but no extractable text. "
        content += f"Type: {result.pdf_type}. "
        if result.pages_needing_ocr:
            content += f"Pages needing OCR: {result.pages_needing_ocr}. "
        content += "Consider uploading a text-based PDF.]"

    return ParsedDocument(
        content=content,
        title=result.title,
        doc_type="pdf",
        page_count=result.page_count,
        metadata={
            "pdf_type": result.pdf_type,
            "confidence": result.confidence,
            "pages_needing_ocr": result.pages_needing_ocr,
            "pages_with_tables": result.pages_with_tables,
            "pages_with_columns": result.pages_with_columns,
            "has_encoding_issues": result.has_encoding_issues,
            "processing_time_ms": result.processing_time_ms,
        },
    )
