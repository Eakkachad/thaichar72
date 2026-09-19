"""Thai character class definitions and utilities for the 72-class TIS-620 corpus."""

import unicodedata

# Hard-coded 72 TIS-620 byte codes, sorted ascending
CLASS_CODES: list[int] = [
    161, 162, 163, 164, 167, 168, 169, 170, 171, 173,
    175, 176, 177, 178, 179, 180, 181, 182, 183, 184,
    185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
    195, 196, 197, 199, 200, 201, 202, 203, 204, 205,
    206, 207, 209, 210, 212, 213, 214, 215, 216, 217,
    224, 225, 226, 227, 228, 229, 230, 231, 232, 233,
    234, 236, 240, 241, 242, 243, 244, 245, 246, 247,
    248, 249,
]

# Build lookup structures
_code_to_index: dict[int, int] = {c: i for i, c in enumerate(CLASS_CODES)}


def code_to_char(code: int) -> str:
    """Convert a TIS-620 byte code to its Thai Unicode character."""
    return chr(0x0E00 + code - 0xA0)


def char_to_code(ch: str) -> int:
    """Convert a Thai character to its TIS-620 byte code."""
    return ord(ch) - 0x0E00 + 0xA0


def code_to_index(code: int) -> int:
    """Return the 0-based class index (0..71) for a TIS-620 code."""
    return _code_to_index[code]


def index_to_code(i: int) -> int:
    """Return the TIS-620 code for a 0-based class index."""
    return CLASS_CODES[i]


# List of Thai characters corresponding to CLASS_CODES
CLASS_CHARS: list[str] = [code_to_char(c) for c in CLASS_CODES]


def category(code: int) -> str:
    """Categorise a TIS-620 code as consonant | vowel | tone_mark | digit."""
    cp = 0x0E00 + code - 0xA0  # Unicode code point
    # Digits: U+0E50 – U+0E59
    if 0x0E50 <= cp <= 0x0E59:
        return "digit"
    # Consonants: U+0E01 – U+0E2E plus ฯ U+0E2F
    if 0x0E01 <= cp <= 0x0E2F:
        return "consonant"
    # Vowels: U+0E30 – U+0E39 and U+0E40 – U+0E45
    if 0x0E30 <= cp <= 0x0E39 or 0x0E40 <= cp <= 0x0E45:
        return "vowel"
    # Tone/diacritic marks: U+0E46 – U+0E4E
    if 0x0E46 <= cp <= 0x0E4E:
        return "tone_mark"
    raise ValueError(f"Code {code} (U+{cp:04X}) does not fit any category")


def unicode_name(code: int) -> str:
    """Return the Unicode character name for a TIS-620 code."""
    return unicodedata.name(code_to_char(code))
