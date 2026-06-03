"""
turkish_plate_regex.py — Turkish license plate format filter & normalizer.

Turkish plate format:
  [2-digit city code] [1-3 uppercase letters] [2-4 digits]
  Examples: "34 ABC 1234", "06 K 47", "35 WEB 001"

This filter:
  1. Strips whitespace and normalizes to uppercase
  2. Applies regex matching
  3. Returns standardized "XX LLL DDDD" string or None if no match
"""
import re
from .auto_correct import OCRAutoCorrector

# Turkish plate regex:
# - 2 digit city code (01-81)
# - 1-3 uppercase letters (Turkish alphabet subset, no Q/X/W officially but we allow all A-Z)
# - 2-4 digits
_PLATE_PATTERN = re.compile(
    r'^(\d{2})\s*([A-Z]{1,3})\s*(\d{2,4})$'
)

# Valid city codes in Turkey: 01–81 (some gaps exist but we allow 01-81)
_MIN_CITY = 1
_MAX_CITY = 81


class TurkishPlateFilter:
    """
    Validates and normalizes Turkish license plate strings.

    Usage:
        f = TurkishPlateFilter()
        result = f.apply("34ABC1234")   # → "34 ABC 1234"
        result = f.apply("invalid")     # → None
    """
    def __init__(self):
        self.corrector = OCRAutoCorrector()

    def apply(self, text: str) -> str | None:
        """
        Clean, validate, and normalize a plate string.

        Args:
            text: Raw OCR output string

        Returns:
            Normalized plate string ("34 ABC 1234") or None if invalid
        """
        if not text:
            return None

        # Normalize: uppercase, strip outer whitespace, collapse inner spaces
        cleaned = text.upper().strip()
        # Remove all whitespace and recheck
        no_space = re.sub(r'\s+', '', cleaned)
        
        # Try to fix OCR errors using contextual rules
        corrected_no_space = self.corrector.apply(no_space)

        match = _PLATE_PATTERN.match(corrected_no_space)
        if not match:
            return None

        city_str, letters, digits = match.groups()
        city_int = int(city_str)
        if not (_MIN_CITY <= city_int <= _MAX_CITY):
            return None

        # Normalize: zero-pad city to 2 digits (already 2 from regex)
        return f"{city_str} {letters} {digits}"

    def apply_batch(self, texts: list[str]) -> list[str | None]:
        """Apply filter to a list of OCR strings."""
        return [self.apply(t) for t in texts]
