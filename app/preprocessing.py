"""
Preprocessing layer: normalizes noisy/poorly-written ticket text before
it hits the classifier. This is the module that absorbs most of the
"30% noisy tickets" constraint from the problem statement.
"""
import re

# Small domain-specific normalization map. In production this would be
# a proper spellchecker (SymSpell / hunspell) trained on IT vocabulary,
# but a lookup table keeps this prototype dependency-light and fast.
COMMON_FIXES = {
    "teh": "the", "adn": "and", "internert": "internet", "passwrod": "password",
    "cant": "cannot", "keps": "keeps", "conection": "connection", "acess": "access",
    "compter": "computer", "prnter": "printer", "pls": "please", "isue": "issue",
    "wont": "will not", "asap": "as soon as possible",
}

FILLER_PATTERN = re.compile(r"\b(asap|plz|help|ticket|urgent)\b", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
NON_ALPHANUMERIC = re.compile(r"[^a-z0-9\s]")


def clean_text(raw_text: str) -> str:
    """Lowercase, fix common typos, strip noise/filler tokens, collapse whitespace."""
    if not raw_text:
        return ""

    text = raw_text.lower().strip()
    text = NON_ALPHANUMERIC.sub(" ", text)

    words = text.split()
    fixed_words = [COMMON_FIXES.get(w, w) for w in words]
    text = " ".join(fixed_words)

    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def is_low_information(text: str, min_tokens: int = 2) -> bool:
    """Flags tickets too short/garbled to classify reliably (e.g. failed OCR)."""
    return len(text.split()) < min_tokens
