"""
SkinSage — Your Personal Beauty Companion
Run:  streamlit run app.py
Needs: streamlit>=1.40, pandas, pillow, requests  (optional: pytesseract)
Put the .streamlit/config.toml file next to app.py to force the light theme.
"""

import html
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

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
    page_title="SkinSage | Your Personal Beauty Companion",
    page_icon="🌸",
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

/* ---------- scanner ---------- */
[data-testid="stFileUploaderDropzone"]{ background:#f7f3ff !important; border:1.5px dashed #b9a5e7 !important;
  border-radius:18px !important; color:#3a3a5c !important; }
[data-testid="stFileUploaderDropzone"] *{ color:#3a3a5c !important; }
[data-testid="stFileUploaderDropzone"] button{ background:#fff !important; border:1px solid #cdbff0 !important; border-radius:12px !important; }
.scanner-box{ background:linear-gradient(135deg,#f4edff,#fff4f8); border:1px dashed #b9a5e7;
  border-radius:22px; padding:26px; text-align:center; }
.scanner-title{ font-family:'Playfair Display',serif; color:#182653; font-size:26px; }
.scanner-text{ color:#62627a; font-size:14px; margin-top:6px; }
textarea{ background:#fff !important; color:#17245d !important; }

.footer{ text-align:center; padding:26px 0 4px; color:#7b7891; font-size:13px; }

@media (max-width: 900px){ .nav{ display:none; } .hero-title{ font-size:38px; } }
</style>
"""
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

# =========================================================
# HERO + PREFERENCE CARD
# =========================================================
hero_l, hero_r = st.columns([0.30, 0.70], gap="large")

with hero_l:
    st.markdown(
        html_block(
            """
            <div class="hero-title">Discover<br>Beauty That<br><span class="pink">Matches You</span></div>
            <div class="hero-desc">Get personalised skincare and makeup recommendations
            based on your skin type, concerns and tone.</div>
            """
        ),
        unsafe_allow_html=True,
    )

with hero_r:
    with st.container(key="pref_card", border=True):
        tab_skin, tab_makeup = st.tabs(["🪷  Skincare Recommendation", "✨  Makeup Recommendation"])

        # ---------------- skincare tab ----------------
        with tab_skin:
            st.markdown('<div class="section-label">Select your skin type</div>', unsafe_allow_html=True)
            with st.container(key="skin_row"):
                for col, (name, icon) in zip(st.columns(5), SKIN_TYPES.items()):
                    col.button(
                        f"{icon} {name}",
                        key=f"skin_{name}",
                        type="primary" if ss.skin_type == name else "secondary",
                        on_click=pick_skin,
                        args=(name,),
                        use_container_width=True,
                    )

            st.markdown(
                '<div class="section-label">Select your concerns <span>(choose any)</span></div>',
                unsafe_allow_html=True,
            )
            with st.container(key="concern_grid"):
                for row_start in range(0, len(CONCERNS), 5):
                    for col, name in zip(st.columns(5), CONCERNS[row_start : row_start + 5]):
                        selected = name in ss.concerns
                        col.button(
                            ("✓ " if selected else "") + name,
                            key=f"concern_{name}",
                            type="primary" if selected else "secondary",
                            on_click=toggle_concern,
                            args=(name,),
                            use_container_width=True,
                        )

            _, mid, _ = st.columns([0.2, 0.6, 0.2])
            with mid:
                if st.button("Get Skincare Recommendations  →", use_container_width=True, key="skin_go"):
                    with st.spinner("Finding your routine..."):
                        ss.skin_groups = rank_products(
                            skin_df, SKIN_CATEGORIES, SKIN_ORDER,
                            skin_type=ss.skin_type, concerns=ss.concerns,
                        )

        # ---------------- makeup tab ----------------
        with tab_makeup:
            m1, m2 = st.columns(2)
            tone = m1.selectbox("Select your skin tone", SKIN_TONES, index=1, key="tone_sel")
            m_type = m2.selectbox("Select your skin type", list(SKIN_TYPES), key="mtype_sel")
            _, mid, _ = st.columns([0.2, 0.6, 0.2])
            with mid:
                if st.button("Get Makeup Recommendations  →", use_container_width=True, key="makeup_tab_go"):
                    with st.spinner("Finding your picks..."):
                        ss.makeup_groups = rank_products(
                            makeup_df, MAKEUP_CATEGORIES, MAKEUP_ORDER, skin_type=m_type, tone=tone,
                        )

# Defaults so the page is never empty on first load
if ss.skin_groups is None:
    ss.skin_groups = rank_products(skin_df, SKIN_CATEGORIES, SKIN_ORDER, skin_type=ss.skin_type)
if ss.makeup_groups is None:
    ss.makeup_groups = rank_products(makeup_df, MAKEUP_CATEGORIES, MAKEUP_ORDER)

# =========================================================
# SKINCARE ROUTINE
# =========================================================
with st.container(key="sec_skin"):
    section_header("🌱", "Your Recommended Skincare Routine",
                   "Based on your skin type and concerns, here are the best products for you.")
    if skin_df.empty:
        st.info("Skincare dataset not found. Expected one of: " + ", ".join(SKIN_FILES))
    else:
        render_groups(ss.skin_groups, SKIN_CATEGORIES, SKIN_ORDER)

# =========================================================
# MAKEUP
# =========================================================
with st.container(key="sec_makeup"):
    section_header("✨", "Makeup Recommendations",
                   "Based on your skin tone and skin type, here are the best picks for you.")
    if makeup_df.empty:
        st.info("Makeup dataset not found. Add a CSV such as makeup_final.csv to enable this section.")
    else:
        render_groups(ss.makeup_groups, MAKEUP_CATEGORIES, MAKEUP_ORDER)

# =========================================================
# PRODUCT SCANNER
# =========================================================
with st.container(key="sec_scan"):
    section_header("🔍", "Scan Your Product", "Upload a photo of your product and SkinSage will try to identify it.")
    sc_l, sc_r = st.columns([0.45, 0.55], gap="large")

    with sc_l:
        upload = st.file_uploader(
            "Upload product image", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed"
        )
        if upload:
            st.image(Image.open(upload), caption="Uploaded product", use_container_width=True)

    with sc_r:
        st.markdown(
            html_block(
                """
                <div class="scanner-box">
                    <div class="scanner-title">Product Scanner</div>
                    <div class="scanner-text">Upload a clear photo of the packaging. SkinSage reads the
                    visible text with OCR and looks for a matching product.</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
        if upload and st.button("🔍 Scan Product", use_container_width=True, key="scan_go"):
            with st.spinner("Scanning product..."):
                text = None
                try:
                    import pytesseract

                    text = pytesseract.image_to_string(Image.open(upload)).strip()
                except ImportError:
                    st.error("OCR isn't installed. Run `pip install pytesseract` and install Tesseract OCR.")
                except Exception as exc:
                    st.error(f"OCR failed: {exc}")

                if text is not None:
                    if not text:
                        st.warning("No readable text found. Try a sharper, well-lit photo.")
                    else:
                        st.text_area("Text found on the product", text, height=140)
                        name_col = find_col(skin_df, ["name", "product name", "product_name", "product", "title"])
                        if name_col:
                            q = " ".join(text.lower().split())
                            ranked = sorted(
                                skin_df[name_col].dropna().astype(str).unique(),
                                key=lambda n: SequenceMatcher(None, q, n.lower()).ratio(),
                                reverse=True,
                            )[:3]
                            st.markdown("**Closest matches in our catalogue**")
                            for n in ranked:
                                st.write("•", clean_name(n, 80))

st.markdown('<div class="footer">SkinSage · AI-assisted skincare and beauty recommendations</div>', unsafe_allow_html=True)