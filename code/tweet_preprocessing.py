import re
import unicodedata

def normalizeTweet(text: str, lower: bool = True) -> str:

    if not isinstance(text, str):
        return ""

    text = re.sub(r'\s+', ' ', text)
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('--', '—')

    if lower:
        text = text.lower()

    text = text.strip()
    return text