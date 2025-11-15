# import re
#
# import nltk
#
# # Download punkt if not available
# try:
#     nltk.data.find("tokenizers/punkt")
# except LookupError:
#     nltk.download("punkt")
#
#
# def preprocess_text(text: str, min_len: int = 10):
#     """
#     Clean raw text and split into sentences.
#     """
#     text = re.sub(r"\s+", " ", text.strip())
#     sentences = nltk.sent_tokenize(text)
#     cleaned = [s.strip() for s in sentences if len(s.strip()) >= min_len]
#     return cleaned


import re

import nltk


def preprocess_text(text, min_len=10):
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = nltk.sent_tokenize(text)

    # Merge short sentences (< 8 words) with the next one
    merged = []
    temp = ""
    for s in sentences:
        if len(s.split()) < 8:
            temp += " " + s
            continue
        if temp:
            merged.append((temp + " " + s).strip())
            temp = ""
        else:
            merged.append(s.strip())
    if temp:
        merged.append(temp.strip())

    cleaned = [s for s in merged if len(s.split()) >= min_len]
    return cleaned
