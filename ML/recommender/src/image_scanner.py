
import io
import re

import pandas as pd
from PIL import Image
from rapidfuzz import process, fuzz


def normalize_text(value):
    value = "" if pd.isna(value) else str(value)
    value = value.lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def extract_text(image_bytes):
    """OCR using Tesseract."""
    try:
        import pytesseract
    except ImportError:
        raise ImportError(
            "pytesseract is not installed. Run: pip install pytesseract"
        )

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return pytesseract.image_to_string(
        image,
        config="--psm 6"
    ).strip()


def match_product(ocr_text, product_df, top_n=5):
    """Match OCR text to brand + name + category in the product dataset."""
    query = normalize_text(ocr_text)

    if not query or product_df.empty:
        return pd.DataFrame()

    search_text = []
    for _, row in product_df.iterrows():
        text = " ".join(
            str(row.get(col, ""))
            for col in ["brand", "name", "label"]
        )
        search_text.append(normalize_text(text))

    matches = process.extract(
        query,
        search_text,
        scorer=fuzz.token_set_ratio,
        limit=top_n,
    )

    rows = []
    for _, score, index in matches:
        row = product_df.iloc[index].copy()
        row["match_score"] = score
        rows.append(row)

    return pd.DataFrame(rows)
