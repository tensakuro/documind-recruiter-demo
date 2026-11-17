# DocuMind AI — Recruiter Edition

A lightweight, free-to-host public demo that showcases document retrieval, evidence inspection, and RAG evaluation metrics.

## What this demo does
- Interactive question-to-passage retrieval over a small sample corpus.
- Shows ranked passages, source pages, similarity scores, and measured retrieval latency.
- Computes Precision@5, Recall@5, MRR, and nDCG@5 against a small labelled benchmark.
- Separately reports historical results from the full local DocuMind pipeline.

## Important technical distinction
The hosted recruiter edition uses TF-IDF + cosine similarity and extractive evidence display. It does **not** call an LLM and does not connect to the full local MySQL, Supabase, Qdrant, BGE, CrossEncoder, or Ollama services. Historical local pipeline results are labelled separately in the dashboard.

## Run locally
Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a new public GitHub repository named `documind-recruiter-demo`.
2. Upload the contents of this folder to the repository root.
3. Visit https://share.streamlit.io/ and sign in with GitHub.
4. Choose the repository, branch `main`, and main file `app.py`.
5. Deploy. No secrets or environment variables are required.

## Metrics
The public benchmark is deliberately small and manually labelled. Its metrics are computed at runtime. They are not statistically conclusive.
The historical local pipeline metrics are from a separate evaluation with 5 questions and 18 labelled chunks:
- Precision@5: 0.4000 → 0.4400
- Recall@5: 0.6133 → 0.6633
- MRR: 0.8000 → 0.8500
- nDCG@5: 0.6501 → 0.6939

## Privacy
The app uses only the sample corpus shipped in `data/`. It has no upload feature, database, secrets, or external model API.
