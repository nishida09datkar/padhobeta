import pytesseract
from PIL import Image
from models.schemas import ParsedDocument


def parse_image(file_path: str) -> ParsedDocument:
    img = Image.open(file_path)

    text = pytesseract.image_to_string(img)

    if not text.strip():
        content = "[No text could be extracted from this image. The image may be too low quality or contain only graphics.]"
    else:
        content = f"[OCR Extracted from Image]\n\n{text}"

    width, height = img.size
    return ParsedDocument(
        content=content,
        title=None,
        doc_type="image",
        page_count=1,
        metadata={
            "image_width": width,
            "image_height": height,
            "format": img.format,
            "mode": img.mode,
        },
    )
