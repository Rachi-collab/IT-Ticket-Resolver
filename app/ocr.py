from io import BytesIO
from typing import Tuple

try:
    import pytesseract
    from PIL import Image

    _OCR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OCR_AVAILABLE = False


def ocr_available() -> bool:
    if not _OCR_AVAILABLE:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def extract_text_from_image(image_bytes: bytes) -> Tuple[str, float]:
    if not ocr_available():
        return "", 0.0

    try:
        image = Image.open(BytesIO(image_bytes)).convert("L")
        text = pytesseract.image_to_string(image)
    except Exception:
        return "", 0.0

    text = text.strip()
    alnum_chars = sum(c.isalnum() for c in text)
    confidence = min(1.0, alnum_chars / 40.0) if text else 0.0
    return text, confidence
