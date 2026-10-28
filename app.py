"""
app.py  —  Banking Intent Classifier  (Streamlit Demo)
=========================================================
Run:
    streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import joblib
import streamlit as st

# ─────────────────────────────────────────────────────────
# Page config  (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Banking Intent Classifier",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# Custom CSS — dark premium look
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Main background */
  .stApp { background: #0f1117; }

  /* Hide default Streamlit header decoration */
  header[data-testid="stHeader"] { background: transparent; }

  /* Cards */
  .card {
    background: #1a1d26;
    border: 1px solid #2d3152;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
  }
  .result-card {
    background: linear-gradient(135deg, #1a1d26 0%, #1e2235 100%);
    border: 1px solid #4F6EF7;
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1.5rem;
  }
  .intent-badge {
    background: linear-gradient(90deg, #4F6EF7, #6d83f7);
    color: white;
    padding: 8px 20px;
    border-radius: 50px;
    font-size: 1.1rem;
    font-weight: 700;
    display: inline-block;
    margin: 0.5rem 0;
    letter-spacing: 0.02em;
  }
  .conf-high  { color: #43C59E; font-weight: 700; font-size: 1.5rem; }
  .conf-med   { color: #F7C948; font-weight: 700; font-size: 1.5rem; }
  .conf-low   { color: #F76F6E; font-weight: 700; font-size: 1.5rem; }
  .alt-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 12px; border-radius: 8px; margin: 4px 0;
    background: #22263a;
  }
  .alt-name  { color: #e8eaf2; font-size: 0.9rem; }
  .alt-score { color: #8a8fa8; font-size: 0.85rem; font-family: monospace; }
  .section-title {
    color: #8a8fa8; font-size: 0.75rem; text-transform: uppercase;
    letter-spacing: 0.1em; font-weight: 600; margin-bottom: 0.5rem;
  }
  .stTextArea textarea {
    background: #1a1d26 !important;
    border: 1px solid #2d3152 !important;
    border-radius: 10px !important;
    color: #e8eaf2 !important;
    font-size: 1rem !important;
  }
  .stButton>button {
    background: linear-gradient(90deg, #4F6EF7, #7c8ff7);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2.5rem;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    transition: transform 0.15s;
  }
  .stButton>button:hover { transform: translateY(-2px); }
  div[data-testid="metric-container"] {
    background: #1a1d26;
    border: 1px solid #2d3152;
    border-radius: 10px;
    padding: 1rem;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    artifact     = joblib.load(BASE_DIR / "mini_jev_choice.joblib")
    model        = artifact["model"]
    category_names = artifact["category_names"]
    return model, category_names

model, category_names = load_model()
classes = model.named_steps["ann"].classes_

def display_label(label):
    return category_names.get(str(label), str(label))

def readable(label):
    """Convert snake_case to Title Case for display."""
    return label.replace("_", " ").replace("?", "").title()

# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 About this app")
    st.markdown("""
This app demonstrates a **Banking Intent Classifier** trained using:
- **TF-IDF** — converts text to numbers
- **MLP Neural Network** — 256 × 128 hidden layers
- **Dataset** — Banking77 (PolyAI, 13K samples)
- **Classes** — 77 banking intent categories

---
### 📊 Model Performance
""")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy",  "87.53%")
        st.metric("Precision", "88.29%")
    with col2:
        st.metric("Recall",    "87.53%")
        st.metric("Macro F1",  "87.65%")

    st.markdown("""
---
### 🔬 Try these examples
""")
    examples = [
        "My transfer has not arrived",
        "I want to activate my card",
        "What is the exchange rate today?",
        "My card was stolen",
        "I forgot my PIN",
        "How do I top up by bank transfer?",
        "My card was swallowed by the ATM",
        "I want to cancel my account",
        "Why is my balance not updated?",
        "I was charged twice for the same transaction",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}"):
            st.session_state["user_input"] = ex

# ─────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1.5rem;">
  <div style="font-size:3rem; margin-bottom:0.5rem">🏦</div>
  <h1 style="color:#e8eaf2; font-size:2rem; font-weight:700; margin:0">
    Banking Intent Classifier
  </h1>
  <p style="color:#8a8fa8; margin-top:0.5rem">
    ANN College Project · Banking77 Dataset · 77 Classes
  </p>
</div>
""", unsafe_allow_html=True)

# Input area
user_input = st.text_area(
    "Enter a customer banking message:",
    value=st.session_state.get("user_input", ""),
    height=100,
    placeholder="e.g. 'My card hasn't arrived yet' or 'I forgot my PIN'",
    key="input_box",
    label_visibility="collapsed",
)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
predict_clicked = st.button("🔍  Classify Intent", use_container_width=True)

# ─────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────
if predict_clicked and user_input.strip():
    with st.spinner("Classifying..."):
        text    = user_input.strip()
        proba   = model.predict_proba([text])[0]
        top5_idx = np.argsort(proba)[::-1][:5]

        best_idx   = top5_idx[0]
        best_label = classes[best_idx]
        best_name  = display_label(best_label)
        best_conf  = float(proba[best_idx])

    # ── Primary result card ──
    conf_class = "conf-high" if best_conf >= 0.7 else ("conf-med" if best_conf >= 0.4 else "conf-low")
    conf_color = "#43C59E" if best_conf >= 0.7 else ("#F7C948" if best_conf >= 0.4 else "#F76F6E")
    conf_emoji = "✅" if best_conf >= 0.7 else ("⚠️" if best_conf >= 0.4 else "❌")

    st.markdown(f"""
<div class="result-card">
  <div class="section-title">Predicted Intent</div>
  <div class="intent-badge">{readable(best_name)}</div>
  <br>
  <div style="display:flex; gap:3rem; margin-top:1rem">
    <div>
      <div class="section-title">Confidence</div>
      <span class="{conf_class}">{conf_emoji} {best_conf:.1%}</span>
    </div>
    <div>
      <div class="section-title">Intent ID</div>
      <code style="color:#a5b4fc; font-size:1rem">{best_name}</code>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Confidence progress bar
    st.progress(best_conf)

    # ── Confidence interpretation ──
    if best_conf >= 0.85:
        st.success("**Very high confidence** — the model is certain about this intent.")
    elif best_conf >= 0.60:
        st.warning("**Moderate confidence** — likely correct but may be similar to another intent.")
    else:
        st.error("**Low confidence** — the message is ambiguous. Check the alternatives below.")

    # ── Alternative predictions ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Alternative Predictions (Top 5)")

    alt_html = ""
    for rank, idx in enumerate(top5_idx):
        label   = classes[idx]
        name    = display_label(label)
        score   = float(proba[idx])
        bar_w   = int(score * 100)
        is_best = rank == 0
        bg      = "background: #2a2f4a;" if is_best else "background: #22263a;"
        border  = "border-left: 3px solid #4F6EF7;" if is_best else ""
        pct_col = "#4F6EF7" if is_best else "#43C59E" if score > 0.2 else "#F7C948"

        alt_html += f"""
<div style="{bg} {border} border-radius:8px; padding:10px 14px; margin:6px 0;
     display:flex; justify-content:space-between; align-items:center">
  <div>
    <span style="color:#e8eaf2; font-size:.9rem; font-weight:{'700' if is_best else '400'}">
      {'🥇 ' if is_best else f'{rank+1}. '}{readable(name)}
    </span><br>
    <code style="color:#8a8fa8; font-size:.75rem">{name}</code>
  </div>
  <div style="text-align:right">
    <span style="color:{pct_col}; font-weight:700; font-size:1rem">{score:.1%}</span>
  </div>
</div>
"""

    st.markdown(alt_html, unsafe_allow_html=True)

    # ── Probability bar chart ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 Confidence Bar Chart")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.5))
    top5_names  = [readable(display_label(classes[i]))[:30] for i in top5_idx]
    top5_scores = [float(proba[i]) for i in top5_idx]
    colors = ["#4F6EF7"] + ["#43C59E"] * 4

    bars = ax.barh(top5_names[::-1], top5_scores[::-1], color=colors[::-1],
                   edgecolor="white", height=0.55)
    for bar, val in zip(bars, top5_scores[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10, fontweight="bold", color="white")
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Probability", fontsize=11, color="white")
    ax.set_title("Top-5 Predicted Intents", fontsize=13, fontweight="bold", color="white")
    ax.tick_params(colors="white")
    ax.spines[["top", "right", "left", "bottom"]].set_color("#2d3152")
    ax.set_facecolor("#1a1d26")
    fig.patch.set_facecolor("#1a1d26")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

elif predict_clicked and not user_input.strip():
    st.warning("Please enter a message before clicking Classify.")

# ─────────────────────────────────────────────────────────
# Batch Testing Section
# ─────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🔬 Batch Test — Try multiple messages at once"):
    batch_input = st.text_area(
        "Enter multiple messages (one per line):",
        height=150,
        placeholder="My card hasn't arrived\nI forgot my PIN\nWhat is the exchange rate?",
        key="batch_input",
    )
    if st.button("Run Batch Prediction", key="batch_btn"):
        messages = [m.strip() for m in batch_input.strip().split("\n") if m.strip()]
        if messages:
            rows = []
            for msg in messages:
                proba = model.predict_proba([msg])[0]
                best_i = int(np.argmax(proba))
                rows.append({
                    "Message"   : msg,
                    "Predicted Intent": readable(display_label(classes[best_i])),
                    "Confidence": f"{float(proba[best_i]):.1%}",
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ─────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#4a4f6a; font-size:12px; margin-top:3rem; padding-top:1rem;
     border-top: 1px solid #2d3152">
  Banking Intent Classifier · ANN College Project · Banking77 Dataset (PolyAI) ·
  TF-IDF + MLP Neural Network · 87.53% Test Accuracy
</div>
""", unsafe_allow_html=True)
