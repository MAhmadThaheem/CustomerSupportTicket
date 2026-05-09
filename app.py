# ============================================================
# HSRIS - Step 5: Streamlit App
# Run locally : streamlit run app.py
# Deploy on   : streamlit.io/cloud
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import re
from collections import Counter
from utils import CustomTokenizer, VocabularyBuilder, LabelEncoderCustom, OneHotEncoderCustom
# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title = "HSRIS - Ticket Retrieval System",
    page_icon  = "app",
    layout     = "wide"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global styling overrides */
    .stApp {
        background: linear-gradient(135deg, #050b14, #0a192f);
        color: #ccd6f6;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    .block-container {
        padding-top: 2rem;
    }

    /* Animations */
    @keyframes pulse-cyan {
        0%, 100% { text-shadow: 0 0 10px #00f0ff, 0 0 20px #00f0ff; }
        50% { text-shadow: 0 0 15px #00f0ff, 0 0 30px #00f0ff; }
    }
    @keyframes float-up {
        0% { opacity: 0; transform: translateY(15px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    /* Headers */
    .main-header {
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0.5rem;
        color: #fff;
        text-transform: uppercase;
        letter-spacing: 3px;
        animation: pulse-cyan 3s infinite ease-in-out;
    }
    .main-header span {
        color: #00f0ff;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #64ffda;
        margin-bottom: 3rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* Ticket Cards */
    .ticket-card {
        background: rgba(10, 25, 47, 0.85);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(0, 240, 255, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        color: #8892b0;
        transition: all 0.3s ease;
        animation: float-up 0.6s ease-out forwards;
        box-shadow: 0 10px 30px -10px rgba(2, 12, 27, 0.7);
    }
    .ticket-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 30px rgba(0, 240, 255, 0.15);
        border-color: #00f0ff;
    }

    .ticket-card-tfidf {
        border-left: 4px solid #00f0ff;
    }
    .ticket-card-glove {
        border-left: 4px solid #64ffda;
    }

    /* Badges */
    .score-badge {
        background: rgba(0, 240, 255, 0.1);
        color: #00f0ff;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 700;
        border: 1px solid rgba(0, 240, 255, 0.3);
        box-shadow: 0 0 10px rgba(0, 240, 255, 0.1);
    }
    .type-badge {
        background: rgba(0, 85, 255, 0.1);
        color: #64ffda;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 700;
        border: 1px solid rgba(100, 255, 218, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Status Colors */
    .priority-high   { color: #ff0055 !important; font-weight: 800; text-shadow: 0 0 5px rgba(255, 0, 85, 0.4); }
    .priority-medium { color: #fdf500 !important; font-weight: 800; text-shadow: 0 0 5px rgba(253, 245, 0, 0.4); }
    .priority-low    { color: #00ff00 !important; font-weight: 800; text-shadow: 0 0 5px rgba(0, 255, 0, 0.4); }
    .text-highlight  { color: #e6f1ff; font-weight: 600; }

    /* Streamlit Metric Overrides */
    [data-testid="stMetricValue"] {
        color: #00f0ff !important;
        text-shadow: 0 0 10px rgba(0, 240, 255, 0.3);
        font-size: 2.2rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #64ffda !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
    }
    [data-testid="metric-container"] {
        background: rgba(10, 25, 47, 0.6);
        border: 1px solid rgba(0, 240, 255, 0.2);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: all 0.3s;
    }
    [data-testid="metric-container"]:hover {
        border-color: #00f0ff;
        box-shadow: inset 0 0 15px rgba(0, 240, 255, 0.1);
    }

    /* Form UI Override */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #00f0ff, #0055ff) !important;
    }
    
    [data-testid="baseButton-primary"] {
        background: linear-gradient(45deg, #0055ff, #00f0ff) !important;
        color: #050b14 !important;
        font-weight: 800 !important;
        border-radius: 8px !important;
        border: none !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.4) !important;
        transition: all 0.3s ease;
    }
    [data-testid="baseButton-primary"]:hover {
        transform: scale(1.02);
        box-shadow: 0 0 25px rgba(0, 240, 255, 0.6) !important;
    }

    [data-testid="baseButton-secondary"] {
        background: transparent !important;
        border: 1px solid rgba(0, 240, 255, 0.3) !important;
        color: #00f0ff !important;
        transition: all 0.3s ease;
    }
    [data-testid="baseButton-secondary"]:hover {
        background: rgba(0, 240, 255, 0.1) !important;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.3) !important;
    }

    .stTextArea textarea {
        background: rgba(2, 12, 27, 0.7) !important;
        color: #e6f1ff !important;
        border: 1px solid rgba(0, 240, 255, 0.3) !important;
        border-radius: 8px !important;
    }
    .stTextArea textarea:focus {
        border-color: #00f0ff !important;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.2) !important;
    }

    hr { border-color: rgba(255, 255, 255, 0.05) !important; }
    h3 {
        color: #ccd6f6 !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    h4 {
        color: #64ffda !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Artifacts (cached) ───────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_all_artifacts():
    """Load all models and data. Cached so it only runs once."""

    df = pd.read_parquet("tickets_clean.parquet")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("vocab_builder.pkl", "rb") as f:
        vocab_builder = pickle.load(f)

    with open("glove_vocab.pkl", "rb") as f:
        glove_vocab = pickle.load(f)

    idf_scores          = np.load("idf_scores.npy")
    tfidf_dense         = torch.load("tfidf_matrix.pt",
                                     map_location=DEVICE)
    glove_matrix_corpus = torch.load("glove_matrix.pt",
                                     map_location=DEVICE)

    # Rebuild GloVe embedding layer from corpus matrix
    # We use the saved matrix directly for lookups
    EMBED_DIM = glove_matrix_corpus.shape[1]

    return (df, tokenizer, vocab_builder, glove_vocab,
            idf_scores, tfidf_dense, glove_matrix_corpus, EMBED_DIM)


# ── Helper Functions ──────────────────────────────────────────
def generate_ngrams(tokens, n):
    if len(tokens) < n:
        return []
    return ["_".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def augment_with_ngrams(token_list):
    augmented = list(token_list)
    augmented += generate_ngrams(token_list, 2)
    augmented += generate_ngrams(token_list, 3)
    return augmented

def encode_query_tfidf(query, tokenizer, vocab, idf_scores):
    tokens = tokenizer.tokenize(query)
    tokens = augment_with_ngrams(tokens)
    counts = Counter(tokens)
    vec    = np.zeros(len(vocab), dtype=np.float32)
    for token, count in counts.items():
        col = vocab.get(token)
        if col is not None:
            vec[col] = count * idf_scores[col]
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec /= norm
    return torch.tensor(vec, dtype=torch.float32, device=DEVICE)

def encode_query_glove(query, tokenizer, glove_vocab,
                        vocab, idf_scores, embed_dim):
    tokens       = tokenizer.tokenize(query)
    unk_idx      = glove_vocab["<UNK>"]
    weighted_sum = torch.zeros(embed_dim, device=DEVICE)
    total_weight = 0.0

    for token in tokens:
        glove_idx  = glove_vocab.get(token, unk_idx)
        tfidf_col  = vocab.get(token)
        weight     = float(idf_scores[tfidf_col]) if tfidf_col is not None else 0.1

        # Direct lookup from glove_vocab index into glove_matrix
        vec = glove_matrix_corpus[glove_idx].to(DEVICE)
        weighted_sum  += weight * vec
        total_weight  += weight

    if total_weight > 1e-12:
        weighted_sum /= total_weight
    norm = torch.norm(weighted_sum)
    if norm > 1e-12:
        weighted_sum = weighted_sum / norm
    return weighted_sum

def hybrid_search(query, alpha, top_k=5):
    """Run hybrid search and return results."""
    tfidf_qvec = encode_query_tfidf(
        query, tokenizer, vocab_builder.vocab, idf_scores)
    glove_qvec = encode_query_glove(
        query, tokenizer, glove_vocab,
        vocab_builder.vocab, idf_scores, EMBED_DIM)

    tfidf_sims  = torch.mv(tfidf_dense, tfidf_qvec)
    glove_sims  = torch.mv(glove_matrix_corpus, glove_qvec)
    final_sims  = alpha * tfidf_sims + (1 - alpha) * glove_sims

    top_k_res   = torch.topk(final_sims, top_k)
    indices     = top_k_res.indices.tolist()
    scores      = top_k_res.values.tolist()

    results = []
    for idx, score in zip(indices, scores):
        results.append({
            "score"       : round(score, 4),
            "tfidf_score" : round(tfidf_sims[idx].item(), 4),
            "glove_score" : round(glove_sims[idx].item(), 4),
            "ticket_type" : df["ticket_type"].iloc[idx],
            "priority"    : df["priority"].iloc[idx],
            "channel"     : df["channel"].iloc[idx],
            "description" : df["description"].iloc[idx],
        })

    # Also get pure TF-IDF and pure GloVe top-3 for comparison
    tfidf_top3 = torch.topk(tfidf_sims, 3).indices.tolist()
    glove_top3 = torch.topk(glove_sims,  3).indices.tolist()

    tfidf_results = [{
        "score"       : round(tfidf_sims[i].item(), 4),
        "ticket_type" : df["ticket_type"].iloc[i],
        "priority"    : df["priority"].iloc[i],
        "description" : df["description"].iloc[i],
    } for i in tfidf_top3]

    glove_results = [{
        "score"       : round(glove_sims[i].item(), 4),
        "ticket_type" : df["ticket_type"].iloc[i],
        "priority"    : df["priority"].iloc[i],
        "description" : df["description"].iloc[i],
    } for i in glove_top3]

    # Predict ticket type by majority vote
    type_votes     = Counter([r["ticket_type"] for r in results])
    predicted_type = type_votes.most_common(1)[0][0]

    return results, tfidf_results, glove_results, predicted_type


def priority_color(priority):
    colors = {"High": "priority-high", "Medium": "priority-medium", "Low": "priority-low"}
    return colors.get(priority, "")


# ── Load Data ─────────────────────────────────────────────────
with st.spinner("Loading models and data..."):
    (df, tokenizer, vocab_builder, glove_vocab,
     idf_scores, tfidf_dense, glove_matrix_corpus, EMBED_DIM) = load_all_artifacts()


# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="main-header">HSRIS <span>NEXUS</span></div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Neural Ticket Retrieval System</div>',
            unsafe_allow_html=True)

# Stats row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Tickets",    f"{len(df):,}")
c2.metric("Vocabulary Size",  "5,000")
c3.metric("GloVe Dimensions", "300")
c4.metric("Ticket Types",     df["ticket_type"].nunique())

st.divider()

# ── Input Section ─────────────────────────────────────────────

col_input, col_controls = st.columns([3, 1])

with col_input:
    st.subheader("Enter a Support Ticket")
    query = st.text_area(
        label       = "Ticket Description",
        placeholder = "e.g. My payment was declined but money was deducted from my account...",
        height      = 120,
        label_visibility = "collapsed"
    )

with col_controls:
    st.markdown("**Alpha (α) — Search Mode**")
    alpha = st.slider(
        label      = "α value",
        min_value  = 0.0,
        max_value  = 1.0,
        value      = 0.4,
        step       = 0.05,
        help       = "0.0 = pure semantic (GloVe) | 1.0 = pure keyword (TF-IDF)",
        label_visibility = "collapsed"
    )

    # Alpha mode label
    if alpha <= 0.2:
        mode_label = "SEMANTIC OVERRIDE"
        mode_color = "#64ffda"
    elif alpha >= 0.8:
        mode_label = "KEYWORD SYNTAX"
        mode_color = "#00f0ff"
    else:
        mode_label = "NEURAL HYBRID"
        mode_color = "#0055ff"

    st.markdown(
        f"<div style='color:{mode_color};font-weight:800;font-size:1rem;text-shadow:0 0 10px {mode_color};margin-bottom:0.5rem;'>"
        f"α={alpha:.2f} — {mode_label}</div>",
        unsafe_allow_html=True
    )


    search_btn = st.button("Search", type="primary", use_container_width=True)


# ── Results ───────────────────────────────────────────────────
if search_btn and query.strip():
    with st.spinner("Searching..."):
        results, tfidf_results, glove_results, predicted_type = hybrid_search(
            query, alpha, top_k=5)

    st.divider()

    # Prediction
    st.markdown(f"### Predicted Ticket Type")
    st.success(f"**{predicted_type}** (majority vote of top-5 results)")

    st.divider()

    # Top 3 Hybrid Results
    st.markdown("### Top 3 Similar Past Resolutions")
    for r in results[:3]:
        prio_cls = priority_color(r["priority"])
        st.markdown(f"""
        <div class="ticket-card" style="display:flex; flex-direction:column; justify-content:space-between;">
            <div>
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
                    <span class="type-badge">{r['ticket_type']}</span>
                    <span class="score-badge">SCORE: {r['score']}</span>
                </div>
                <div style="font-size:1.0rem;color:#e6f1ff;margin-bottom:16px;line-height:1.6;text-align:justify;">{r['description'][:200]}...</div>
            </div>
            <div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; font-size:0.85rem; color:#8892b0; border-top:1px solid rgba(0, 240, 255, 0.2); padding-top:12px; margin-top:auto;">
                <div>Priority: <span class="{prio_cls}">{r['priority']}</span> &nbsp;|&nbsp; Channel: <span class="text-highlight">{r['channel']}</span></div>
                <div>TF-IDF: <span class="text-highlight">{r['tfidf_score']}</span> &nbsp;|&nbsp; GloVe: <span class="text-highlight">{r['glove_score']}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Side-by-side TF-IDF vs GloVe
    st.markdown("### TF-IDF vs GloVe Comparison")
    st.caption("See how keyword vs semantic search differ on the same query")

    col_tfidf, col_glove = st.columns(2)

    with col_tfidf:
        st.markdown("#### TF-IDF (KEYWORD SYNTAX)")
        for i, r in enumerate(tfidf_results, 1):
            st.markdown(f"""
            <div class="ticket-card ticket-card-tfidf" style="animation-delay: {i*0.1}s; display:flex; flex-direction:column; justify-content:space-between; height:180px;">
                <div>
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                        <span class="type-badge" style="background:rgba(0, 240, 255, 0.1); color:#00f0ff; border:1px solid rgba(0, 240, 255, 0.3); box-shadow:0 0 10px rgba(0, 240, 255, 0.2);">{r['ticket_type']}</span>
                        <span style="color:#00f0ff;font-weight:800;text-shadow:0 0 5px #00f0ff;font-size:0.9rem;">Score: {r['score']}</span>
                    </div>
                    <div style="font-size:0.9rem;color:#e6f1ff;line-height:1.5;text-align:justify;">{r['description'][:120]}...</div>
                </div>
                <div style="font-size:0.8rem; border-top:1px solid rgba(0, 240, 255, 0.2); padding-top:8px; text-align:right; margin-top:auto; color:#64ffda;">
                   Priority: <span class="{priority_color(r['priority'])}">{r['priority']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_glove:
        st.markdown("#### GLOVE (SEMANTIC NODE)")
        for i, r in enumerate(glove_results, 1):
            st.markdown(f"""
            <div class="ticket-card ticket-card-glove" style="animation-delay: {i*0.1}s; display:flex; flex-direction:column; justify-content:space-between; height:180px;">
                <div>
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                        <span class="type-badge" style="background:rgba(100, 255, 218, 0.1); color:#64ffda; border:1px solid rgba(100, 255, 218, 0.3); box-shadow:0 0 10px rgba(100, 255, 218, 0.2);">{r['ticket_type']}</span>
                        <span style="color:#64ffda;font-weight:800;text-shadow:0 0 5px #64ffda;font-size:0.9rem;">Score: {r['score']}</span>
                    </div>
                    <div style="font-size:0.9rem;color:#e6f1ff;line-height:1.5;text-align:justify;">{r['description'][:120]}...</div>
                </div>
                <div style="font-size:0.8rem; border-top:1px solid rgba(100, 255, 218, 0.2); padding-top:8px; text-align:right; margin-top:auto; color:#00f0ff;">
                   Priority: <span class="{priority_color(r['priority'])}">{r['priority']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

elif search_btn and not query.strip():
    st.warning("Please enter a ticket description to search.")


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### How it works")
    st.markdown("""
    **Formula:**
    ```
    Score = α × TF-IDF + (1-α) × GloVe
    ```
    **TF-IDF** finds tickets with matching keywords.

    **GloVe** finds tickets with similar *meaning* — e.g. "money" matches "billing".

    **Alpha slider** lets you control the balance between the two approaches.
    """)

    st.markdown("### Example Queries")
    examples = [
        "payment declined but money taken",
        "cannot login to my account",
        "product arrived broken",
        "I want a refund",
        "app keeps crashing",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state["example_query"] = ex

    st.markdown("---")
    st.markdown("Build by M Ahmad")
