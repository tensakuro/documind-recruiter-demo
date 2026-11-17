import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
chunks = json.loads((ROOT / "data" / "chunks.json").read_text())
benchmarks = json.loads((ROOT / "data" / "benchmark.json").read_text())

st.set_page_config(page_title="DocuMind AI | Recruiter Demo", page_icon="📄", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1180px;}
.hero {padding: 1.4rem 1.6rem; border: 1px solid rgba(128,128,128,.25); border-radius: 16px; margin-bottom: 1rem;}
.muted {color: #8b949e;}
.small {font-size: .85rem; color: #8b949e;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def build_index():
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix

vectorizer, matrix = build_index()

def retrieve(question, top_k=5):
    started = time.perf_counter()
    qv = vectorizer.transform([question])
    scores = cosine_similarity(qv, matrix).ravel()
    order = np.argsort(scores)[::-1][:top_k]
    elapsed_ms = (time.perf_counter() - started) * 1000
    results = []
    for idx in order:
        item = dict(chunks[int(idx)])
        item["score"] = float(scores[idx])
        results.append(item)
    return results, elapsed_ms

def tokenize_count(text):
    return len(re.findall(r"\b[\w'-]+\b", text))

def precision_recall_at_k(retrieved_ids, relevant_ids, k=5):
    top = retrieved_ids[:k]
    relevant = set(relevant_ids)
    hits = sum(1 for x in top if x in relevant)
    precision = hits / k
    recall = hits / len(relevant) if relevant else 0.0
    return precision, recall

def reciprocal_rank(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved_ids, relevant_ids, k=5):
    relevant = set(relevant_ids)
    gains = [1 if x in relevant else 0 for x in retrieved_ids[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(gains, reverse=True)
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0

def run_benchmark():
    rows = []
    for item in benchmarks:
        results, _ = retrieve(item["question"], top_k=5)
        ids = [x["id"] for x in results]
        p, r = precision_recall_at_k(ids, item["relevant"], 5)
        rows.append({
            "Precision@5": p,
            "Recall@5": r,
            "MRR": reciprocal_rank(ids, item["relevant"]),
            "nDCG@5": ndcg_at_k(ids, item["relevant"], 5),
        })
    return pd.DataFrame(rows).mean().to_dict()

st.markdown("""
<div class="hero">
  <div class="small">PUBLIC PORTFOLIO DEMO · DOCUMENT INTELLIGENCE</div>
  <h1 style="margin:.2rem 0;">📄 DocuMind AI</h1>
  <p style="font-size:1.05rem;margin-bottom:.2rem;">Ask questions. Retrieve evidence. Inspect the metrics.</p>
  <p class="muted">A lightweight, reproducible retrieval demo built to showcase the RAG workflow and evaluation methodology.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("DocuMind AI")
    st.caption("Recruiter edition")
    page = st.radio("Navigate", ["RAG Playground", "Evaluation Dashboard", "Architecture & Engineering"])
    st.divider()
    st.caption("Demo corpus: 12 passages · 6 pages")
    st.caption("Retrieval: TF-IDF + cosine similarity")
    st.caption("No external API or API key required.")

if page == "RAG Playground":
    st.subheader("RAG playground")
    st.write("Try a question about the sample RAG engineering document.")
    examples = [
        "What are the main stages of a RAG pipeline?",
        "Why is reranking useful after vector search?",
        "How should a RAG application measure latency?",
        "How can multi-user retrieval be secured?",
    ]
    selected = st.selectbox("Example questions", ["Choose an example..."] + examples)
    default_q = "" if selected == "Choose an example..." else selected
    question = st.text_input("Your question", value=default_q, max_chars=500, placeholder="e.g. How do MRR and nDCG differ?")
    top_k = st.slider("Number of passages to retrieve (k)", 1, 8, 5)
    if st.button("Retrieve evidence", type="primary", disabled=not question.strip()):
        results, retrieval_ms = retrieve(question.strip(), top_k=top_k)
        st.session_state["last_query"] = question.strip()
        st.session_state["last_results"] = results
        st.session_state["last_retrieval_ms"] = retrieval_ms
    if "last_results" in st.session_state:
        results = st.session_state["last_results"]
        retrieval_ms = st.session_state["last_retrieval_ms"]
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Retrieval latency", f"{retrieval_ms:.2f} ms")
        c2.metric("Retrieved passages", len(results))
        c3.metric("Query tokens", tokenize_count(st.session_state["last_query"]))
        st.subheader("Evidence-based response")
        positive = [x for x in results if x["score"] > 0]
        if not positive:
            st.info("No matching passages were found. Try a more specific question.")
        else:
            # Transparent extractive synthesis: show the strongest evidence rather than inventing an LLM answer.
            st.write("The strongest matching evidence indicates:")
            for item in positive[:3]:
                st.markdown(f"**{item['section']}** — {item['text']}")
                st.caption(f"Source: Sample RAG Engineering Guide · Page {item['page']} · Chunk {item['id']}")
        st.subheader("Retrieved passages")
        for rank, item in enumerate(results, start=1):
            with st.expander(f"#{rank} · {item['section']} · Page {item['page']} · Score {item['score']:.4f}", expanded=(rank == 1)):
                st.write(item["text"])
                st.caption(f"Chunk ID: {item['id']} · TF-IDF cosine similarity: {item['score']:.4f}")
        st.caption("Method note: this public edition uses lexical TF-IDF retrieval and extractive evidence display. It does not call an LLM. The full local DocuMind pipeline is documented separately.")

elif page == "Evaluation Dashboard":
    st.subheader("Retrieval evaluation")
    st.write("The benchmark uses seven curated questions with manually labelled relevant passage IDs. Metrics are computed live against the demo's retrieval output.")
    scores = run_benchmark()
    cols = st.columns(4)
    for col, key in zip(cols, ["Precision@5", "Recall@5", "MRR", "nDCG@5"]):
        col.metric(key, f"{scores[key]:.3f}")
    st.caption("These are results of this lightweight public demo benchmark. They are not the same experiment as the full local BGE/Qdrant/CrossEncoder evaluation.")
    rows = []
    for item in benchmarks:
        res, ms = retrieve(item["question"], 5)
        ids = [x["id"] for x in res]
        p, r = precision_recall_at_k(ids, item["relevant"], 5)
        rows.append({"Question": item["question"], "P@5": p, "R@5": r, "MRR": reciprocal_rank(ids, item["relevant"]), "nDCG@5": ndcg_at_k(ids, item["relevant"], 5), "Latency (ms)": round(ms, 2)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.subheader("Full local pipeline — previously measured evaluation")
    st.caption("Separate experiment: 5 questions and 18 labelled chunks. These historical results belong to the full local pipeline, not this TF-IDF public demo.")
    historical = pd.DataFrame([
        {"Metric":"Precision@5","Before reranking":0.4000,"After reranking":0.4400},
        {"Metric":"Recall@5","Before reranking":0.6133,"After reranking":0.6633},
        {"Metric":"MRR","Before reranking":0.8000,"After reranking":0.8500},
        {"Metric":"nDCG@5","Before reranking":0.6501,"After reranking":0.6939},
    ])
    st.dataframe(historical, use_container_width=True, hide_index=True)
    st.bar_chart(historical.set_index("Metric"))
    st.info("Small evaluation sample: interpret these values as project-level experimental results, not statistically conclusive benchmarks.")

else:
    st.subheader("Architecture & engineering")
    st.markdown("""
    **Full local DocuMind pipeline**
    1. PDF upload and validation
    2. Text extraction and cleaning
    3. Token-aware chunking
    4. BGE-small embeddings
    5. Qdrant vector indexing with user/document/version filters
    6. Candidate retrieval (top 10)
    7. CrossEncoder reranking (top 5)
    8. Ollama local answer generation
    9. Source citation validation and latency instrumentation

    **Core stack**
    - Python · FastAPI · Pydantic · SQLAlchemy · MySQL
    - Qdrant · Sentence Transformers · Hugging Face
    - CrossEncoder reranking · Ollama
    - Supabase private object storage
    """)
    st.warning("The hosted recruiter edition is intentionally lightweight: TF-IDF retrieval over a fixed sample corpus. It does not connect to your private database, Supabase bucket, local Qdrant instance, or local Ollama server.")
    st.markdown("**Engineering focus:** owner-scoped retrieval, versioned indexes, processing-attempt ownership, retryable deletion, migration testing, and retrieval evaluation.")
    st.caption("GitHub repository link: add your repository URL here after publishing this demo.")

st.divider()
st.caption("DocuMind AI · Recruiter Edition · Free, sample-data-only demonstration · No user documents are stored.")
