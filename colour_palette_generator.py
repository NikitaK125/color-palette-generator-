# ============================================================
#   COLOUR PALETTE GENERATOR USING MACHINE LEARNING
#   Nikita Kalbande | ML Portfolio Project - Special
#   Algorithm: K-Means Clustering
#   Run with: streamlit run colour_palette_generator.py
# ============================================================

import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Colour Palette Generator",
    page_icon="🎨",
    layout="wide"
)

# ── CSS — exact match to the mockup UI ───────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=DM+Sans:wght@300;400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #F5F0E8 !important;
}
.main, .block-container {
    background-color: #F5F0E8 !important;
    padding-top: 0 !important;
    max-width: 100% !important;
}

/* Navbar */
.navbar {
    background: #ffffff;
    border-bottom: 0.5px solid #E0D8CC;
    padding: 12px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: -1rem -1rem 1.5rem -1rem;
}
.navbar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 15px;
    font-weight: 500;
    color: #2C2C2A;
}
.navbar-sub {
    font-size: 11px;
    color: #999;
}

/* Sidebar overrides */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 0.5px solid #E0D8CC;
}
[data-testid="stSidebar"] .block-container {
    background-color: #ffffff !important;
    padding-top: 1.5rem !important;
}

/* Upload box */
.upload-box {
    border: 1.5px dashed #C8B99A;
    border-radius: 12px;
    background: #FBF8F3;
    padding: 40px 20px;
    text-align: center;
    cursor: pointer;
}
.upload-icon { font-size: 36px; }
.upload-title { font-size: 14px; font-weight: 500; color: #2C2C2A; margin: 8px 0 4px; }
.upload-sub   { font-size: 12px; color: #aaa; }
.upload-hint  { font-size: 12px; color: #D8944C; margin-top: 8px; }

/* Swatch row */
.swatch-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
    border-bottom: 0.5px solid #F0EBE0;
}
.swatch-box {
    width: 46px;
    height: 46px;
    border-radius: 8px;
    border: 0.5px solid rgba(0,0,0,0.07);
    flex-shrink: 0;
}
.swatch-name { font-size: 13px; font-weight: 500; color: #2C2C2A; }
.swatch-hex  { font-family: monospace; font-size: 11px; color: #999; margin-top: 2px; }

/* Palette strip card */
.palette-strip-card {
    background: #ffffff;
    border: 0.5px solid #E0D8CC;
    border-radius: 12px;
    padding: 16px;
    margin-top: 8px;
}
.strip-swatch {
    text-align: center;
}
.strip-block {
    height: 64px;
    border-radius: 8px;
    border: 0.5px solid rgba(0,0,0,0.06);
    margin-bottom: 6px;
}
.strip-name { font-size: 11px; font-weight: 500; color: #2C2C2A; }
.strip-hex  { font-family: monospace; font-size: 10px; color: #999; }

/* Hex code box */
.hex-box {
    background: #ffffff;
    border: 0.5px solid #E0D8CC;
    border-radius: 10px;
    padding: 14px 16px;
}
.hex-box-label { font-size: 11px; color: #999; margin-bottom: 6px; }
.hex-box-codes {
    font-family: monospace;
    font-size: 12px;
    color: #2C2C2A;
    background: #F5F0E8;
    padding: 8px 12px;
    border-radius: 6px;
    word-break: break-all;
}

/* Download button */
[data-testid="stDownloadButton"] > button {
    background-color: #D8944C !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    width: 100% !important;
    padding: 0.55rem 1rem !important;
    font-size: 14px !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background-color: #67482C !important;
}

/* File uploader tweak */
[data-testid="stFileUploader"] {
    background: #FBF8F3;
    border: 1.5px dashed #C8B99A;
    border-radius: 12px;
    padding: 10px;
}

/* How it works steps */
.step-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 10px;
}
.step-num {
    width: 22px; height: 22px;
    border-radius: 50%;
    background: #F5F0E8;
    border: 0.5px solid #E0D8CC;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 500; color: #D8944C;
    flex-shrink: 0;
}
.step-text { font-size: 12px; color: #666; padding-top: 2px; }

/* Tag pills */
.tag {
    display: inline-block;
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 4px;
    background: #F5F0E8;
    color: #999;
    border: 0.5px solid #E0D8CC;
    margin: 2px;
}

/* Section title */
.sec-title { font-size: 13px; font-weight: 500; color: #2C2C2A; margin-bottom: 10px; }

/* Divider */
.divider { border: none; border-top: 0.5px solid #E0D8CC; margin: 12px 0; }

/* Footer */
.footer { text-align: center; color: #bbb; font-size: 11px; margin-top: 32px; padding-top: 16px; border-top: 0.5px solid #E0D8CC; }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────

def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def get_colour_name(hex_code):
    r = int(hex_code[1:3], 16)
    g = int(hex_code[3:5], 16)
    b = int(hex_code[5:7], 16)
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    lightness = (max_c + min_c) / 2

    if max_c == min_c:
        if lightness < 50:    return "Charcoal"
        elif lightness < 150: return "Grey"
        else:                 return "White"
    if r > g and r > b:
        if r > 200 and g > 100: return "Sandy Orange"
        if r > 150 and g < 100: return "Terracotta"
        return "Earthy Red"
    elif g > r and g > b:
        if g > 150 and r > 100: return "Olive Green"
        return "Botanical Green"
    elif b > r and b > g:
        if b > 150 and g > 100: return "Slate Blue"
        return "Deep Teal"
    elif r > 150 and g > 150:
        if b < 100: return "Warm Ochre"
        return "Warm Cream"
    elif r > 100 and b > 100:
        return "Dusty Mauve"
    return "Deep Clay"

def extract_palette(image, n_colors=6):
    img = image.resize((200, 200))
    img_array = np.array(img)
    pixels = img_array.reshape(-1, 3).astype(float)
    mask = (pixels.sum(axis=1) > 60) & (pixels.sum(axis=1) < 720)
    filtered = pixels[mask]
    if len(filtered) < n_colors:
        filtered = pixels
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(filtered)
    colours = kmeans.cluster_centers_
    counts  = np.bincount(kmeans.labels_)
    colours = colours[np.argsort(-counts)]
    return colours

def create_palette_image(colours):
    n = len(colours)
    fig, ax = plt.subplots(figsize=(n * 2, 3))
    fig.patch.set_facecolor("#F5F0E8")
    for i, c in enumerate(colours):
        hex_c = rgb_to_hex(c)
        rect  = mpatches.FancyBboxPatch(
            (i * 2 + 0.1, 0.3), 1.8, 2.2,
            boxstyle="round,pad=0.05",
            facecolor=[x/255 for x in c],
            edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        ax.text(i * 2 + 1, 0.1, hex_c,
                ha="center", va="center",
                fontsize=8, color="#2C2C2A",
                fontfamily="monospace", fontweight="bold")
    ax.set_xlim(0, n * 2)
    ax.set_ylim(0, 2.8)
    ax.axis("off")
    plt.tight_layout(pad=0.2)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor="#F5F0E8")
    plt.close()
    buf.seek(0)
    return buf


# ── Navbar ────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">
        🎨 &nbsp; Colour Palette Generator
    </div>
    <span class="navbar-sub">by Nikita Kalbande · @storiesbynikita</span>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sec-title">Number of colours</div>', unsafe_allow_html=True)
    n_colors = st.slider("", min_value=3, max_value=10, value=6, label_visibility="collapsed")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">How it works</div>', unsafe_allow_html=True)
    steps = [
        "Upload any photo or moodboard",
        "K-Means ML groups similar pixel colours",
        "Get hex codes, RGB values & colour names",
        "Download palette or copy hex codes",
    ]
    for i, s in enumerate(steps, 1):
        st.markdown(f"""
        <div class="step-row">
            <div class="step-num">{i}</div>
            <div class="step-text">{s}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Built with</div>', unsafe_allow_html=True)
    st.markdown("""
    <div>
        <span class="tag">Scikit-learn</span>
        <span class="tag">K-Means</span>
        <span class="tag">Streamlit</span>
        <span class="tag">Pillow</span>
        <span class="tag">NumPy</span>
        <span class="tag">Matplotlib</span>
    </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px; color:#bbb;">Made by Nikita Kalbande<br>@storiesbynikita</div>',
                unsafe_allow_html=True)


# ── Main content ──────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload your image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    with st.spinner("🎨 Analysing colours with ML..."):
        colours = extract_palette(image, n_colors)

    # ── Two columns: image | swatches ────────────────────────
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.markdown('<div class="sec-title">Your image</div>', unsafe_allow_html=True)
        st.image(image, use_column_width=True)

    with col2:
        st.markdown('<div class="sec-title">Extracted palette</div>', unsafe_allow_html=True)
        for c in colours:
            hex_c = rgb_to_hex(c)
            name  = get_colour_name(hex_c)
            r, g, b = int(c[0]), int(c[1]), int(c[2])
            st.markdown(f"""
            <div class="swatch-row">
                <div class="swatch-box" style="background:{hex_c};"></div>
                <div>
                    <div class="swatch-name">{name}</div>
                    <div class="swatch-hex">{hex_c} &nbsp;·&nbsp; rgb({r},{g},{b})</div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Palette strip ─────────────────────────────────────────
    st.markdown('<div class="sec-title">Your palette</div>', unsafe_allow_html=True)
    strip_cols = st.columns(n_colors)
    for i, c in enumerate(colours):
        hex_c = rgb_to_hex(c)
        name  = get_colour_name(hex_c)
        with strip_cols[i]:
            st.markdown(f"""
            <div class="strip-swatch">
                <div class="strip-block" style="background:{hex_c};"></div>
                <div class="strip-name">{name}</div>
                <div class="strip-hex">{hex_c}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Hex codes + Download ──────────────────────────────────
    bot1, bot2 = st.columns([2, 1], gap="medium")

    with bot1:
        hex_line = "  ·  ".join([rgb_to_hex(c) for c in colours])
        st.markdown(f"""
        <div class="hex-box">
            <div class="hex-box-label">Copy hex codes</div>
            <div class="hex-box-codes">{hex_line}</div>
        </div>""", unsafe_allow_html=True)

    with bot2:
        palette_img = create_palette_image(colours)
        st.download_button(
            label="⬇️  Download Palette Image",
            data=palette_img,
            file_name="nikita_palette.png",
            mime="image/png",
            use_container_width=True
        )

else:
    # ── Empty state ───────────────────────────────────────────
    st.markdown("""
    <div class="upload-box">
        <div class="upload-icon">🖼️</div>
        <div class="upload-title">Upload an image to get started</div>
        <div class="upload-sub">JPG · PNG · WEBP supported</div>
        <div class="upload-hint">Try one of your @storiesbynikita moodboards!</div>
    </div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Colour Palette Generator &nbsp;·&nbsp; Nikita Kalbande &nbsp;·&nbsp; ML Portfolio Project<br>
    Built with K-Means Clustering · Scikit-learn · Streamlit
</div>""", unsafe_allow_html=True)
