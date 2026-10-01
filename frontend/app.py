"""
SkinSage — Your Personal Beauty Companion
Run:  streamlit run app.py
Needs: streamlit>=1.40, pandas, pillow, requests  (optional: pytesseract)
Put the .streamlit/config.toml file next to app.py to force the light theme.
"""

import html
import pickle
import re
import base64
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
from product_matcher import find_product
import textwrap
from ingredient_checker import analyze_ingredients


import pandas as pd
import streamlit as st
from PIL import Image

try:
    import requests
except ImportError:  # app still runs, just without live product images
    requests = None

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="MedSynapse | Your Personal Beauty Companion",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
ML_DIR = BASE_DIR.parent / "ML" / "recommender"
MODEL_PATH = ML_DIR / "models" / "recommender.pkl"
PROCESSED_DIR = ML_DIR / "data" / "processed"

SKIN_FILES = [
    "general_skin_care_final.csv",
    "general_skin_care.csv",
    "general_skincare.csv",
    "general_test_final.csv",
]
MAKEUP_FILES = ["makeup_final.csv", "makeup.csv", "general_makeup.csv", "makeup_products.csv"]

NYKAA_BASE = "https://www.nykaa.com"

# =========================================================
# CONSTANTS
# =========================================================
SKIN_TYPES = {"Normal": "🍃", "Dry": "💧", "Oily": "✦", "Combination": "◯", "Acne-Prone": "✧"}

CONCERNS = [
    "Acne", "Fine Lines", "Wrinkles", "Pigmentation", "Blackheads",
    "Whiteheads", "Blemishes", "Dark Circles", "Eye Bags", "Dark Spots",
]
CONCERN_ALIASES = {
    "Acne": ["acne", "blemish", "pimple"],
    "Fine Lines": ["fine line"],
    "Wrinkles": ["wrinkle", "anti-aging", "anti aging"],
    "Pigmentation": ["pigment"],
    "Blackheads": ["blackhead"],
    "Whiteheads": ["whitehead"],
    "Blemishes": ["blemish"],
    "Dark Circles": ["dark circle"],
    "Eye Bags": ["eye bag", "puffiness"],
    "Dark Spots": ["dark spot"],
}
SKIN_TONES = ["Light", "Light to Medium", "Medium", "Medium to Deep", "Dark to Deep"]

# name -> (icon, subtitle, colour class, keywords). Order matters: first match wins.
SKIN_CATEGORIES = {
    "Eye Cream": ("👁️", "For bright and youthful eyes", "lav", ["eye"]),
    "Mask / Peel": ("🎭", "For deep treatment and exfoliation", "lav", ["mask", "peel", "exfol", "scrub"]),
    "Cleanser": ("🧴", "For deep cleansing and clear skin", "sun", ["cleans", "face wash", "facewash", "wash"]),
    "Moisturiser": ("🫙", "For hydration and skin barrier care", "sun", ["moistur", "cream", "lotion", "gel", "hydrat"]),
}
SKIN_ORDER = ["Cleanser", "Moisturiser", "Mask / Peel", "Eye Cream"]

MAKEUP_CATEGORIES = {
    "Foundation": ("🧴", "Perfect coverage for your skin tone", "sun", ["foundation"]),
    "Concealer": ("💄", "For flawless and even skin", "rose", ["conceal"]),
    "Primer": ("🧪", "For smooth and long-lasting makeup", "lav", ["primer"]),
}
MAKEUP_ORDER = ["Foundation", "Concealer", "Primer"]

# =========================================================
# HELPERS
# =========================================================
def norm(value) -> str:
    return "" if pd.isna(value) else str(value).lower().strip()


def find_col(df: pd.DataFrame, names):
    lookup = {str(c).lower().strip(): c for c in df.columns}
    for n in names:
        if n.lower() in lookup:
            return lookup[n.lower()]
    return None


def first_existing(files):
    for f in files:
        p = PROCESSED_DIR / f
        if p.exists():
            try:
                df = pd.read_csv(p)
                if not df.empty:
                    return df
            except Exception:
                continue
    return pd.DataFrame()


def html_block(s: str) -> str:
    """Strip indentation so Markdown never treats the HTML as a code block."""
    return "\n".join(line.strip() for line in s.splitlines() if line.strip())


def fmt_price(value) -> str:
    text = norm(value)
    if not text:
        return ""
    raw = text.replace("₹", "").replace("rs.", "").replace("rs", "").replace(",", "").strip()
    try:
        return f"₹{float(raw):,.0f}"
    except ValueError:
        return str(value)


def clean_name(name: str, limit: int = 58) -> str:
    name = " ".join(name.split()).title()
    return name if len(name) <= limit else name[: limit - 1].rstrip() + "…"


def normalize_url(url: str) -> str:
    url = url.strip()
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/"):
        return NYKAA_BASE + url
    if url.startswith("nykaa.com") or url.startswith("www.nykaa.com"):
        return "https://" + url
    return url


# =========================================================
# DATA + MODEL
# =========================================================
@st.cache_data(show_spinner=False)
def load_skin():
    return first_existing(SKIN_FILES)


@st.cache_data(show_spinner=False)
def load_makeup():
    return first_existing(MAKEUP_FILES)


@st.cache_resource(show_spinner=False)
def load_model():
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


skin_df = load_skin()
makeup_df = load_makeup()
model = load_model()  # reserved for ML-based ranking

# =========================================================
# NYKAA IMAGE FETCHING
# =========================================================
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
}
_OG_PATTERNS = [
    r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)',
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
    r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)',
]


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_product_image(url: str) -> str:
    """Read the product photo (og:image) from a Nykaa product page. Returns '' on failure."""
    if requests is None or not url.startswith("http"):
        return ""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=8)
        if not resp.ok:
            return ""
        for pattern in _OG_PATTERNS:
            m = re.search(pattern, resp.text, flags=re.I)
            if m:
                return html.unescape(m.group(1))
    except Exception:
        pass
    return ""


def attach_images(groups: dict) -> dict:
    """Fill in missing product images from each product's link, in parallel."""
    todo = [p for items in groups.values() for p in items if not p["image"].startswith("http")]
    if todo:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for p, img in zip(todo, pool.map(lambda x: fetch_product_image(x["url"]), todo)):
                p["image"] = img
    return groups


# =========================================================
# RECOMMENDATION LOGIC
# =========================================================
def classify(text: str, categories: dict):
    for name, (_, _, _, keywords) in categories.items():
        if any(k in text for k in keywords):
            return name
    return None


def product_fields(row: pd.Series) -> dict:
    df = pd.DataFrame([row])

    def get(names):
        col = find_col(df, names)
        return "" if col is None or pd.isna(row[col]) else str(row[col])

    return {
        "name": get(["name", "product name", "product_name", "product", "title"]) or "Beauty Product",
        "price": fmt_price(get(["price", "mrp", "selling price", "selling_price"])),
        "image": get(["image", "image url", "image_url", "product image", "img"]),
        "url": normalize_url(
            get(["url", "product url", "product_url", "link", "product link", "nykaa url", "href"])
        ),
        "label": get(["label", "category", "type"]),
    }


def rank_products(df, categories, order, skin_type=None, concerns=None, tone=None, per_cat=2):
    if df.empty:
        return {}

    concerns = concerns or []
    skin_col = find_col(df, ["skin type", "skin_type", "skintype"])
    concern_col = find_col(df, ["concern", "concerns", "skin concern"])
    tone_col = find_col(df, ["skin tone", "skin_tone", "tone", "shade"])

    scored = []
    for _, row in df.iterrows():
        f = product_fields(row)
        text = f"{f['label']} {f['name']}".lower()
        cat = classify(text, categories)
        if cat is None:
            continue

        score = 0.0
        row_skin = norm(row[skin_col]) if skin_col else ""
        row_concern = norm(row[concern_col]) if concern_col else ""
        row_tone = norm(row[tone_col]) if tone_col else ""

        if skin_type:
            if skin_type.lower() in row_skin:
                score += 5
            elif "all" in row_skin:
                score += 3
        for concern in concerns:
            words = CONCERN_ALIASES.get(concern, [concern.lower()])
            if any(w in row_concern or w in text for w in words):
                score += 3
        if tone and tone_col and tone.lower() in row_tone:
            score += 4
        if f["url"].startswith("http"):
            score += 0.5  # products we can link to (and fetch images for) come first
        scored.append((cat, score, f))

    groups = {}
    for cat in order:
        items = sorted((x for x in scored if x[0] == cat), key=lambda x: -x[1])
        groups[cat] = [x[2] for x in items[:per_cat]]
    return attach_images(groups)


# =========================================================
# HTML RENDERERS
# =========================================================
def product_html(p: dict) -> str:
    name = html.escape(clean_name(p["name"]))
    if p["image"].startswith("http"):
        img = (
            f'<div class="p-img"><img src="{html.escape(p["image"], quote=True)}" '
            f'alt="{name}" referrerpolicy="no-referrer" loading="lazy"></div>'
        )
    else:
        img = '<div class="p-img p-img-empty">🌸</div>'
    price = f'<div class="p-price">{html.escape(p["price"])}</div>' if p["price"] else '<div class="p-price">&nbsp;</div>'
    if p["url"].startswith("http"):
        btn = f'<a class="p-btn" href="{html.escape(p["url"], quote=True)}" target="_blank" rel="noopener">View Details</a>'
    else:
        btn = '<span class="p-btn p-btn-off">View Details</span>'
    return f'<div class="p-card">{img}<div class="p-name">{name}</div>{price}{btn}</div>'


def category_html(name, meta, products) -> str:
    icon, subtitle, tone, _ = meta
    inner = (
        "".join(product_html(p) for p in products)
        if products
        else '<div class="p-empty">No matching products yet.</div>'
    )
    return html_block(
        f"""
        <div class="cat cat-{tone}">
            <div class="cat-head">
                <div class="cat-icon">{icon}</div>
                <div class="cat-text">
                    <div class="cat-title">{html.escape(name)}</div>
                    <div class="cat-sub">{html.escape(subtitle)}</div>
                </div>
            </div>
            <div class="cat-body">{inner}</div>
        </div>
        """
    )


def render_groups(groups, categories, order):
    cols = st.columns(len(order), gap="medium")
    for col, name in zip(cols, order):
        with col:
            st.markdown(category_html(name, categories[name], groups.get(name, [])), unsafe_allow_html=True)


def section_header(icon, title, subtitle):
    st.markdown(
        html_block(
            f"""
            <div class="section-title">{icon} {title}</div>
            <div class="section-sub">{subtitle}</div>
            """
        ),
        unsafe_allow_html=True,
    )


# =========================================================
# CSS
# =========================================================
from pathlib import Path
import base64

# =========================================================
# CSS
# =========================================================
from pathlib import Path
import base64

image_path = Path(__file__).parent / "assets" / "image.jpg"

with open(image_path, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

:root{ color-scheme: light; --ink:#10204f; --ink-soft:#344b98; --violet:#866fe4; --pink:#e26799; }
html, body, [class*="css"], .stApp{ font-family:'DM Sans',sans-serif; color:var(--ink); }
.stApp{
  background:
    radial-gradient(circle at 8% 12%, rgba(219,202,255,.55), transparent 30%),
    radial-gradient(circle at 92% 8%, rgba(255,220,229,.6), transparent 30%),
    linear-gradient(135deg,#fbf8ff 0%,#fff8fb 45%,#f5fcf8 100%) !important;
}
.block-container{ padding:1rem 2.5rem 2rem !important; max-width:1500px; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], .stDeployButton{ display:none !important; }
header[data-testid="stHeader"]{ background:transparent !important; height:0; }
[data-testid="stVerticalBlock"]{ gap:.9rem; }

/* ---------- navbar ---------- */
.navbar{ display:flex; align-items:center; justify-content:space-between;
  padding:4px 4px 16px; border-bottom:1px solid rgba(91,72,145,.12); }
.logo{ display:flex; align-items:center; gap:12px; }
.logo-mark{ width:46px; height:46px; border-radius:50%; display:flex; align-items:center;
  justify-content:center; font-size:24px; background:linear-gradient(135deg,#e6d8ff,#f7c9df); }
.logo-text{ font-family:'Playfair Display',serif; font-size:32px; font-weight:700; color:#6a55c9; line-height:1; }
.logo-text b{ color:var(--pink); font-weight:700; }
.tagline{ font-size:12px; color:#55537a; margin-top:4px; }
.nav{ display:flex; align-items:center; gap:32px; font-size:14px; color:var(--ink); }
.nav .active{ background:#e8ddff; padding:11px 22px; border-radius:24px; }
.nav .login{ background:linear-gradient(135deg,#8875e7,#9d8ff0); color:#fff; padding:12px 32px;
  border-radius:26px; font-weight:600; }

/* ---------- hero ---------- */
.hero-title{ font-family:'Playfair Display',serif; font-size:52px; line-height:1.06;
  font-weight:700; color:var(--ink); margin-top:34px; }
.hero-title .pink{ color:var(--pink); }
.hero-desc{ color:var(--ink-soft); font-size:17px; line-height:1.55; margin-top:18px; max-width:340px; }

/* ---------- preference card ---------- */
.st-key-pref_card{ background:rgba(255,255,255,.92); border:1px solid rgba(120,100,180,.16) !important;
  border-radius:24px !important; padding:14px 26px 22px; box-shadow:0 20px 50px rgba(104,83,170,.10); }
.section-label{ font-size:15px; font-weight:700; color:#162451; margin:10px 0 2px; }
.section-label span{ font-weight:400; }

/* tabs */
.stTabs [data-baseweb="tab-list"]{ gap:8px; border-bottom:0 !important; }
.stTabs [data-baseweb="tab"]{ background:#fff0f3 !important; color:#37364a !important; border-radius:16px !important;
  padding:10px 30px; height:auto; }
.stTabs [data-baseweb="tab"] p{ color:inherit !important; font-weight:500; }
.stTabs [aria-selected="true"]{ background:#f0e4ff !important; color:#12195c !important; }
.stTabs [aria-selected="true"] p{ font-weight:700; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{ display:none !important; }

/* selector buttons (skin type + concerns) */
.st-key-skin_row button, .st-key-concern_grid button{
  height:42px; min-height:42px; padding:0 8px; border-radius:14px !important; font-size:13px !important;
  white-space:nowrap; box-shadow:none !important; transition:background .15s, border-color .15s; }
.st-key-skin_row button[data-testid="stBaseButton-secondary"],
.st-key-concern_grid button[data-testid="stBaseButton-secondary"]{
  background:#fbfafc !important; color:#454457 !important; border:1px solid #e2dfea !important; }
.st-key-skin_row button[data-testid="stBaseButton-secondary"]:hover,
.st-key-concern_grid button[data-testid="stBaseButton-secondary"]:hover{
  border-color:#b9a5e7 !important; background:#f7f2ff !important; color:#3a2d8f !important; }
.st-key-skin_row button[data-testid="stBaseButton-primary"],
.st-key-concern_grid button[data-testid="stBaseButton-primary"]{
  background:#efe6ff !important; color:#3a2d8f !important; border:2px solid #9279e7 !important; font-weight:700; }
.st-key-skin_row p, .st-key-concern_grid p{ font-size:13px; margin:0; color:inherit !important; }
.st-key-skin_row [data-testid="stVerticalBlock"], .st-key-concern_grid [data-testid="stVerticalBlock"]{ gap:.5rem; }

/* call-to-action buttons */
.st-key-skin_go button, .st-key-makeup_tab_go button, .st-key-scan_go button{
  border-radius:30px !important; border:none !important; height:48px; color:#fff !important; font-weight:600;
  background:linear-gradient(90deg,#866fe4,#967be5) !important; box-shadow:0 8px 20px rgba(117,91,203,.22) !important; }
.st-key-skin_go button p, .st-key-makeup_tab_go button p, .st-key-scan_go button p{ color:#fff !important; }

/* select boxes */
[data-baseweb="select"] > div{ background:#fff !important; border-radius:12px !important; border:1px solid #e2dfea !important; }
[data-baseweb="select"] *{ color:#17245d !important; }
label[data-testid="stWidgetLabel"] p{ color:#162451 !important; font-weight:600; }

/* ---------- sections (one bordered container each) ---------- */
.st-key-sec_skin, .st-key-sec_makeup, .st-key-sec_scan{
  border-radius:26px; padding:22px 24px 24px; margin-top:18px; }
.st-key-sec_skin{ background:rgba(244,251,246,.9); border:1px solid rgba(116,180,150,.25); }
.st-key-sec_makeup, .st-key-sec_scan{ background:rgba(255,244,247,.92); border:1px solid rgba(235,160,185,.28); }
.section-title{ font-family:'Playfair Display',serif; font-size:29px; color:#132653; font-weight:700; line-height:1.2; }
.section-sub{ color:#5f6a72; font-size:14px; margin-top:2px; }

/* ---------- category + product cards ---------- */
.cat{ border-radius:20px; padding:14px; border:1px solid rgba(0,0,0,.04); }
.cat-sun{ background:#fff8e6; } .cat-lav{ background:#f1eaff; } .cat-rose{ background:#ffecf1; }
.cat-head{ display:flex; gap:12px; align-items:center; padding:2px 2px 14px; }
.cat-icon{ width:42px; height:42px; border-radius:50%; background:rgba(255,255,255,.85);
  display:flex; align-items:center; justify-content:center; font-size:20px; flex:none; }
.cat-text{ min-width:0; }
.cat-title{ font-family:'Playfair Display',serif; font-size:19px; font-weight:700; color:#18234e; line-height:1.15; }
.cat-sub{ font-size:12px; color:#6b6a80; margin-top:2px; }
.cat-body{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
.p-card{ background:#fff; border-radius:15px; padding:10px; text-align:center; min-width:0;
  display:flex; flex-direction:column; box-shadow:0 4px 14px rgba(80,70,110,.07); }
.p-img{ height:130px; border-radius:12px; background:#fafafa; display:flex; align-items:center; justify-content:center; overflow:hidden; }
.p-img img{ max-width:100%; max-height:100%; object-fit:contain; }
.p-img-empty{ font-size:42px; }
.p-name{ font-size:12.5px; font-weight:500; color:#18234e; margin-top:9px; line-height:1.3; height:49px;
  display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; }
.p-price{ font-size:13px; font-weight:600; color:#252541; margin-top:6px; }
.p-btn{ display:block; margin-top:auto; padding:7px 0; border-radius:18px; font-size:12px; text-decoration:none !important; }
.p-card .p-price + .p-btn{ margin-top:10px; }
.cat-sun .p-btn{ background:#fdeec8; color:#8d5510 !important; }
.cat-lav .p-btn{ background:#e9e0ff; color:#5b46b8 !important; }
.cat-rose .p-btn{ background:#ffdfe8; color:#c2437a !important; }
.p-btn-off{ opacity:.5; }
.p-empty{ grid-column:1/-1; padding:30px 8px; text-align:center; color:#7b7891; font-size:13px; }

/* ---------- premium scanner ---------- */
.st-key-sec_scan{
  background:linear-gradient(120deg,rgba(255,247,250,.98),rgba(247,240,255,.98)) !important;
  border:1px solid rgba(225,145,176,.24) !important;
  border-radius:28px !important;
  padding:20px 24px 18px !important;
  margin-top:18px;
  box-shadow:0 18px 50px rgba(104,83,170,.08);
}
.scan-kicker{display:inline-flex;align-items:center;gap:7px;padding:7px 13px;border-radius:18px;
  background:#eee4ff;color:#5b46bd;font-size:12px;font-weight:700;letter-spacing:.3px;text-transform:uppercase;}
.scan-title{font-family:'Playfair Display',serif;font-size:40px;line-height:1.05;font-weight:700;color:#142451;margin:10px 0 8px;}
.scan-title .pink{color:#e26799;}
.scan-desc{font-size:14px;line-height:1.55;color:#59627c;max-width:520px;margin-bottom:16px;}
.scan-upload-title{font-size:13px;font-weight:700;color:#172451;margin:0 0 7px;}
.scan-benefits{display:flex;gap:22px;margin-top:16px;flex-wrap:wrap;}
.scan-benefit{display:flex;align-items:center;gap:8px;font-size:11px;color:#5e6279;}
.scan-benefit b{display:block;color:#172451;font-size:12px;}
.scan-benefit .ico{width:32px;height:32px;border-radius:50%;background:#fff;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 12px rgba(80,70,110,.08);}
 
.scanner-box{
  width:100%;
  height:320px;
  min-height:320px;
  box-sizing:border-box;
  margin:22px 0 0;          /* pushes card down to match the upload box top */

  background:
    linear-gradient(
      90deg,
      rgba(255,247,250,.98) 0%,
      rgba(255,247,250,.90) 38%,
      rgba(255,247,250,.42) 66%,
      rgba(255,247,250,.08) 100%
    ),
    url("data:image/jpeg;base64,IMAGE_BASE64_PLACEHOLDER");

  background-size:cover;
  background-position:center right;
  background-repeat:no-repeat;

  border:1px solid rgba(164,134,231,.23);
  border-radius:22px;
  padding:28px 30px;

  display:flex;
  flex-direction:column;
  justify-content:center;
  position:relative;
  overflow:hidden;
}
.scanner-box:after{content:'✦';position:absolute;right:22px;top:16px;font-size:27px;color:#9075e8;}
.scanner-title{
  font-family:'Playfair Display',serif;
  color:#182653;
  font-size:28px;
  font-weight:700;
  position:relative;
  z-index:2;
  max-width:55%;
}
.scanner-text{
  color:#62627a;
  font-size:13px;
  line-height:1.55;
  margin-top:8px;
  max-width:52%;
  position:relative;
  z-index:2;
}

.st-key-recommendation_section {
    background: rgba(255,255,255,.9);
    border: 1px solid rgba(120,100,180,.16);
    border-radius: 32px;
    padding: 30px 36px;
    margin-top: 25px;
    box-shadow: 0 18px 50px rgba(104,83,170,.10);
}
/* =========================================================
   PRODUCT UPLOAD BOX - FIXED LAYOUT
   Icon (top) -> Text (middle) -> Button (bottom)
   Each piece is positioned explicitly so nothing can drift.
   ========================================================= */
[data-testid="stFileUploader"]{
  width:100% !important;
  max-width:100% !important;
  margin:0 !important;
}

[data-testid="stFileUploaderDropzone"]{
  position:relative !important;
  display:block !important;              /* no flex flow to fight with */
  width:100% !important;
  height:320px !important;
  min-height:320px !important;
  max-height:320px !important;
  box-sizing:border-box !important;
  padding:0 !important;
  margin:0 !important;
  overflow:hidden !important;
  text-align:center !important;

  background:
    url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='70' height='90' viewBox='0 0 70 90'><path d='M12 82 C8 50 20 20 42 6 C52 32 46 62 12 82Z' fill='%23cfe3c8' opacity='.85'/><path d='M12 82 C20 56 30 34 42 6' stroke='%239fbf95' stroke-width='1.5' fill='none'/></svg>") no-repeat 14px calc(100% - 10px) / 56px auto,
    linear-gradient(180deg,#f8f4ff 0%,#fbf8ff 100%) !important;
  border:2px dashed #b9a5e7 !important;
  border-radius:22px !important;
}

/* inner section wrapper: just a pass-through */
[data-testid="stFileUploaderDropzone"] section{
  position:static !important;
  padding:0 !important;
  margin:0 !important;
}

/* 1) CLOUD ICON - top center */
[data-testid="stFileUploaderDropzone"]::before{
  content:"";
  position:absolute;
  top:28px;
  left:50%;
  transform:translateX(-50%);
  width:64px; height:64px;
  border-radius:50%;
  background:
    url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='34' height='34' viewBox='0 0 24 24' fill='none' stroke='%237b63e0' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M7 18a4.5 4.5 0 0 1-.5-8.97A6 6 0 0 1 18 10.5 4 4 0 0 1 17.5 18'/><path d='M12 19v-7'/><path d='M9 14.5l3-3 3 3'/></svg>") center / 34px no-repeat,
    #e6dcff;
}

/* 2) TEXT BLOCK - directly under the icon */
[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"]{
  position:absolute !important;
  top:108px !important;                  /* 28 icon offset + 64 icon + 16 gap */
  left:0 !important;
  right:0 !important;
  width:100% !important;
  height:auto !important;
  margin:0 !important;
  padding:0 !important;
  display:flex !important;
  flex-direction:column !important;
  align-items:center !important;
  justify-content:flex-start !important;
  gap:4px !important;
  text-align:center !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > *{ display:none !important; }
[data-testid="stFileUploaderDropzoneInstructions"]::before{
  content:"Drag & drop an image here";
  font-size:16px; font-weight:700; color:#172451 !important;
  text-align:center;
}
[data-testid="stFileUploaderDropzoneInstructions"]::after{
  content:"or click to upload\A PNG, JPG, WEBP (max 200MB)";
  white-space:pre-line;
  text-align:center;
  font-size:12px; line-height:1.6; color:#6a6d85 !important;
}

/* 3) BUTTON - pinned to bottom center */
[data-testid="stFileUploaderDropzone"] button{
  position:absolute !important;
  left:50% !important;
  bottom:24px !important;
  top:auto !important;
  transform:translateX(-50%) !important;
  margin:0 !important;
  padding:12px 30px !important;
  border:none !important;
  border-radius:30px !important;
  background:linear-gradient(90deg,#7b63e0,#9a84f0) !important;
  box-shadow:0 10px 22px rgba(123,99,224,.30) !important;
  font-size:0 !important;
  display:flex !important;
  align-items:center !important;
  gap:8px !important;
  white-space:nowrap !important;
  z-index:5 !important;
}
[data-testid="stFileUploaderDropzone"] button *{ color:#fff !important; fill:#fff !important; }
[data-testid="stFileUploaderDropzone"] button p{ display:none !important; }
[data-testid="stFileUploaderDropzone"] button::after{
  content:"Upload Image";
  font-size:15px; font-weight:600; color:#fff !important;
}

/* General text */
[data-testid="stFileUploaderDropzone"] *{
  color:#3a3a5c !important;
}

textarea{
  background:#fff !important;
  color:#17245d !important;
}

@media (max-width:900px){
  [data-testid="stFileUploaderDropzone"]{
    width:100% !important;
    max-width:300px !important;
    margin-left:0 !important;
    margin-right:auto !important;
  }
}
.footer{ text-align:center; padding:26px 0 4px; color:#7b7891; font-size:13px; }

@media (max-width: 900px){ .nav{ display:none; } .hero-title{ font-size:38px; } }
</style>
"""

# Inject the local JPG into the CSS background.
CSS = CSS.replace("IMAGE_BASE64_PLACEHOLDER", image_base64)
st.markdown(CSS, unsafe_allow_html=True)

# =========================================================
# SESSION STATE + CALLBACKS
# =========================================================
ss = st.session_state
ss.setdefault("skin_type", "Normal")
ss.setdefault("concerns", [])
ss.setdefault("skin_groups", None)
ss.setdefault("makeup_groups", None)


def pick_skin(name):
    ss.skin_type = name


def toggle_concern(name):
    if name in ss.concerns:
        ss.concerns.remove(name)
    else:
        ss.concerns.append(name)


# =========================================================
# NAVBAR
# =========================================================
# Layout order: Scan Your Product first, recommendations below.

st.markdown(
    html_block(
        """
        <div class="navbar">
            <div class="logo">
                <div class="logo-mark">🪷</div>
                <div>
                    <div class="logo-text">Skin<b>Sage</b></div>
                    <div class="tagline">Your Personal Beauty Companion</div>
                </div>
            </div>
            <div class="nav">
                <span class="active">Home</span><span>About</span><span>How It Works</span>
                <span>Skin Care</span><span>Makeup</span><span>Contact</span>
                <span style="font-size:20px">⌕</span><span class="login">Login</span>
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

# Defaults so the page is never empty on first load
if ss.skin_groups is None:
    ss.skin_groups = rank_products(skin_df, SKIN_CATEGORIES, SKIN_ORDER, skin_type=ss.skin_type)
if ss.makeup_groups is None:
    ss.makeup_groups = rank_products(makeup_df, MAKEUP_CATEGORIES, MAKEUP_ORDER)

# =========================================================
# PRODUCT SCANNER
# =========================================================

with st.container(key="sec_scan"):

    c_text, c_up, c_card = st.columns(
        [0.34, 0.30, 0.36],
        gap="large"
    )

    # -----------------------------------------------------
    # LEFT: SCANNER INTRO
    # -----------------------------------------------------

    with c_text:

        st.markdown(
            html_block(
                """
                <div class="scan-kicker">
                    ✦ AI Product Scanner
                </div>

                <div class="scan-title">
                    Scan Your <span class="pink">Product</span>
                </div>

                <div class="scan-desc">
                    Upload a photo of your product and SkinSage will try to identify it,
                    read visible packaging text, and help you understand the product.
                </div>

                <div class="scan-benefits">

                    <div class="scan-benefit">
                        <span class="ico">⌕</span>
                        <span>
                            <b>AI Powered</b><br>
                            OCR + smart matching
                        </span>
                    </div>

                    <div class="scan-benefit">
                        <span class="ico">▤</span>
                        <span>
                            <b>Read Ingredients</b><br>
                            Claims & product info
                        </span>
                    </div>

                    <div class="scan-benefit">
                        <span class="ico">♡</span>
                        <span>
                            <b>Personalised</b><br>
                            Skin compatibility
                        </span>
                    </div>

                </div>
                """
            ),
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # MIDDLE: UPLOAD
    # -----------------------------------------------------

    with c_up:

        st.markdown(
            '<div class="scan-upload-title">Upload product image</div>',
            unsafe_allow_html=True
        )

        upload = st.file_uploader(
            "Upload product image",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
            key="product_scanner_upload"
        )


    # -----------------------------------------------------
    # RIGHT: SCANNER CARD
    # -----------------------------------------------------

    with c_card:

        st.markdown(
            html_block(
                """
                <div class="scanner-box">

                    <div class="scanner-title">
                        ✦ Product Scanner
                    </div>

                    <div class="scanner-text">
                        Upload a clear photo of the packaging. SkinSage reads the
                        visible text with OCR and uses AI image matching to identify
                        the closest product in your catalogue.
                    </div>

                </div>
                """
            ),
            unsafe_allow_html=True
        )


# =========================================================
# SCANNER PROCESSING
# IMPORTANT: OUTSIDE sec_scan
# =========================================================

ingredient_result = None
matched_product = None
text = ""

if upload:

    with st.spinner("Scanning product..."):

        # =================================================
        # OCR
        # =================================================

        try:

            import pytesseract

            upload.seek(0)

            image = Image.open(
                upload
            ).convert("RGB")

            text = pytesseract.image_to_string(
                image,
                config="--psm 6"
            ).strip()

            st.write("DEBUG OCR TEXT:")
            st.code(text)

        except ImportError:

            st.error(
                "OCR isn't installed. Run "
                "`pip install pytesseract` and install Tesseract OCR."
            )

        except Exception as exc:

            st.error(
                f"OCR failed: {exc}"
            )


        # =================================================
        # CLIP PRODUCT MATCHING
        # =================================================

        try:

            # OCR has already read the uploaded file.
            # Reset pointer before CLIP.
            upload.seek(0)

            matched_product = find_product(
                upload
            )

        except Exception as exc:

            st.error(
                f"AI product matching failed: {exc}"
            )

            matched_product = None


        # =================================================
        # INGREDIENT ANALYSIS FROM CSV
        # =================================================

        if matched_product:

            csv_ingredients = matched_product.get(
                "ingredients",
                ""
            )

            if (
                csv_ingredients
                and str(csv_ingredients).strip()
            ):

                try:

                    ingredient_result = analyze_ingredients(
                        str(csv_ingredients)
                    )

                except Exception as exc:

                    st.error(
                        f"Ingredient analysis failed: {exc}"
                    )


    # =====================================================
    # OCR RESULT
    # =====================================================

    if text:

        st.markdown(
            "### Text found on the product"
        )

        st.text_area(
            "Detected packaging text",
            text,
            height=140,
            label_visibility="collapsed"
        )

    else:

        st.info(
            "No readable text detected. "
            "AI image matching is being used instead."
        )


# =====================================================
# INGREDIENT ANALYSIS RESULT
# =====================================================

if ingredient_result:

    st.markdown(
        "### Ingredient Analysis"
    )

    ingredient_text = ingredient_result.get(
        "ingredient_text",
        ""
    )

    flagged = ingredient_result.get(
        "flagged",
        []
    )

    # -------------------------------------------------
    # SHOW INGREDIENTS
    # -------------------------------------------------

    if ingredient_text:

        st.markdown(
            "#### Ingredients detected"
        )

        st.text_area(
            "Ingredient text",
            ingredient_text,
            height=150,
            label_visibility="collapsed"
        )

    # -------------------------------------------------
    # SHOW FLAGGED INGREDIENTS
    # -------------------------------------------------

    if flagged:

        st.markdown(
            "#### Ingredients to Review"
        )

        for item in flagged:

            ingredient_name = item.get(
                "ingredient",
                "Unknown ingredient"
            )

            reason = item.get(
                "reason",
                "May require attention."
            )

            level = item.get(
                "level",
                "Caution"
            )

            st.warning(
                f"**{ingredient_name}** — "
                f"{reason} "
                f"({level})"
            )

    elif ingredient_text:

        st.success(
            "No ingredients requiring review "
            "were detected."
        )

    # =====================================================
    # CLIP RESULT
    # =====================================================

    if matched_product:

        similarity = float(
            matched_product.get(
                "similarity",
                0
            )
        )

        status = matched_product.get(
            "status"
        )


        # -------------------------------------------------
        # PRODUCT NOT FOUND
        # -------------------------------------------------

        if status == "Product not found":

            st.warning(
                f"No matching product found in our catalogue. "
                f"AI similarity: {similarity:.2%}"
            )


        # -------------------------------------------------
        # PRODUCT FOUND
        # -------------------------------------------------

        else:

            st.success(
                "Product identified successfully!"
            )

            st.markdown(
                "### AI Matched Product"
            )

            display_items = []

            for key, value in matched_product.items():

                if key in [
                    "similarity",
                    "status"
                ]:
                    continue

                try:

                    if pd.isna(value):
                        continue

                except Exception:

                    pass

                if str(value).strip() == "":
                    continue

                display_items.append(
                    (
                        key,
                        value
                    )
                )


            # -------------------------------------------------
            # PRODUCT DETAILS
            # -------------------------------------------------

            product_cols = st.columns(2)

            for i, (key, value) in enumerate(
                display_items
            ):

                with product_cols[i % 2]:

                    label = (
                        str(key)
                        .replace("_", " ")
                        .title()
                    )

                    st.markdown(
                        html_block(
                            f"""
                            <div class="product-result-item">

                                <div class="product-result-label">
                                    {label}
                                </div>

                                <div class="product-result-value">
                                    {str(value)}
                                </div>

                            </div>
                            """
                        ),
                        unsafe_allow_html=True
                    )


            # -------------------------------------------------
            # AI MATCH SCORE
            # -------------------------------------------------

            st.markdown(
                html_block(
                    f"""
                    <div class="match-score">

                        <b>AI Match Similarity</b>

                        <span>
                            {similarity:.2%}
                        </span>

                    </div>
                    """
                ),
                unsafe_allow_html=True
            )


# =========================================================
# HERO + PREFERENCE CARD
# =========================================================

with st.container(
    key="recommendation_section"
):

    hero_l, hero_r = st.columns(
        [0.30, 0.70],
        gap="medium"
    )


    # -----------------------------------------------------
    # HERO TEXT
    # -----------------------------------------------------

    with hero_l:

        st.markdown(
            html_block(
                """
                <div class="hero-title">
                    Discover<br>
                    Beauty That<br>
                    <span class="pink">Matches You</span>
                </div>

                <div class="hero-desc">
                    Get personalised skincare and makeup recommendations
                    based on your skin type, concerns and tone.
                </div>
                """
            ),
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # PREFERENCE CARD
    # -----------------------------------------------------

    with hero_r:

        with st.container(
            key="pref_card",
            border=True
        ):

            tab_skin, tab_makeup = st.tabs(
                [
                    "🪷  Skincare Recommendation",
                    "✨  Makeup Recommendation"
                ]
            )


            # =================================================
            # SKINCARE TAB
            # =================================================

            with tab_skin:

                st.markdown(
                    '<div class="section-label">'
                    'Select your skin type'
                    '</div>',
                    unsafe_allow_html=True
                )


                # -------------------------------------------------
                # SKIN TYPES
                # -------------------------------------------------

                with st.container(
                    key="skin_row"
                ):

                    skin_columns = st.columns(5)

                    for col, (name, icon) in zip(
                        skin_columns,
                        SKIN_TYPES.items()
                    ):

                        col.button(
                            f"{icon} {name}",
                            key=f"skin_{name}",
                            type=(
                                "primary"
                                if ss.skin_type == name
                                else "secondary"
                            ),
                            on_click=pick_skin,
                            args=(name,),
                            use_container_width=True
                        )


                # -------------------------------------------------
                # CONCERNS
                # -------------------------------------------------

                st.markdown(
                    '<div class="section-label">'
                    'Select your concerns '
                    '<span>(choose any)</span>'
                    '</div>',
                    unsafe_allow_html=True
                )


                with st.container(
                    key="concern_grid"
                ):

                    for row_start in range(
                        0,
                        len(CONCERNS),
                        5
                    ):

                        concern_columns = st.columns(5)

                        for col, name in zip(
                            concern_columns,
                            CONCERNS[
                                row_start:row_start + 5
                            ]
                        ):

                            selected = (
                                name in ss.concerns
                            )

                            col.button(
                                (
                                    "✓ "
                                    if selected
                                    else ""
                                ) + name,

                                key=f"concern_{name}",

                                type=(
                                    "primary"
                                    if selected
                                    else "secondary"
                                ),

                                on_click=toggle_concern,

                                args=(name,),

                                use_container_width=True
                            )


                # -------------------------------------------------
                # RECOMMENDATION BUTTON
                # -------------------------------------------------

                _, mid, _ = st.columns(
                    [0.20, 0.60, 0.20]
                )

                with mid:

                    if st.button(
                        "Get Skincare Recommendations  →",
                        use_container_width=True,
                        key="skin_go"
                    ):

                        with st.spinner(
                            "Finding your routine..."
                        ):

                            ss.skin_groups = rank_products(
                                skin_df,
                                SKIN_CATEGORIES,
                                SKIN_ORDER,
                                skin_type=ss.skin_type,
                                concerns=ss.concerns
                            )


            # =================================================
            # MAKEUP TAB
            # =================================================

            with tab_makeup:

                m1, m2 = st.columns(2)


                # -------------------------------------------------
                # SKIN TONE
                # -------------------------------------------------

                with m1:

                    tone = st.selectbox(
                        "Select your skin tone",
                        SKIN_TONES,
                        index=1,
                        key="tone_sel"
                    )


                # -------------------------------------------------
                # SKIN TYPE
                # -------------------------------------------------

                with m2:

                    m_type = st.selectbox(
                        "Select your skin type",
                        list(SKIN_TYPES),
                        key="mtype_sel"
                    )


                # -------------------------------------------------
                # MAKEUP RECOMMENDATION BUTTON
                # -------------------------------------------------

                _, mid, _ = st.columns(
                    [0.20, 0.60, 0.20]
                )

                with mid:

                    if st.button(
                        "Get Makeup Recommendations  →",
                        use_container_width=True,
                        key="makeup_tab_go"
                    ):

                        with st.spinner(
                            "Finding your picks..."
                        ):

                            ss.makeup_groups = rank_products(
                                makeup_df,
                                MAKEUP_CATEGORIES,
                                MAKEUP_ORDER,
                                skin_type=m_type,
                                tone=tone
                            )


# =========================================================
# SKINCARE ROUTINE
# =========================================================

with st.container(
    key="sec_skin"
):

    section_header(
        "🌱",
        "Your Recommended Skincare Routine",
        "Based on your skin type and concerns, here are the best products for you.",
    )

    if skin_df.empty:

        st.info(
            "Skincare dataset not found. Expected one of: "
            + ", ".join(SKIN_FILES)
        )

    else:

        render_groups(
            ss.skin_groups,
            SKIN_CATEGORIES,
            SKIN_ORDER
        )


# =========================================================
# MAKEUP
# =========================================================

with st.container(
    key="sec_makeup"
):

    section_header(
        "✨",
        "Makeup Recommendations",
        "Based on your skin tone and skin type, here are the best picks for you.",
    )

    if makeup_df.empty:

        st.info(
            "Makeup dataset not found. Add a CSV such as makeup_final.csv "
            "to enable this section."
        )

    else:

        render_groups(
            ss.makeup_groups,
            MAKEUP_CATEGORIES,
            MAKEUP_ORDER
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    'SkinSage · AI-assisted skincare and beauty recommendations'
    '</div>',
    unsafe_allow_html=True
)