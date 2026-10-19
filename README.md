# RAG: Baseline Retrieval vs. Cross-Encoder Re-Ranking

Compares two retrieval pipelines over the same chunked legal document:

1. **Baseline:** Bi-Encoder (`all-MiniLM-L6-v2`) + FAISS → Top 10 → Top 3 to the LLM
2. **Re-ranking:** same Top 10 → Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) → re-sorted Top 3 to the LLM

Both use Gemini (Flash) with an identical prompt. All code is in **`rag_baseline_vs_rerank.ipynb`**.

## Document

Air India — *Conditions of Carriage for Passengers, Baggage and Cargo* (27 pages).
The PDF is not included in this repo. Download it from
<https://www.airindia.com/content/dam/air-india/pdfs/air-india-coc.pdf> (the site blocks scripted downloads, so use a browser) and save it as `data/air-india-coc.pdf`.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your Gemini API key in .env
jupyter notebook rag_baseline_vs_rerank.ipynb
```

`cache/llm_answers.json` holds the Gemini answers from the recorded run, so re-running the notebook reproduces them without an API key or quota. Delete it to regenerate.

## Notebook sections

1. Setup & configuration
2. PDF loading (header and table-of-contents noise removed)
3. Chunking (150 words, 30-word overlap → 105 chunks)
4. Bi-Encoder + FAISS index
5. Cross-Encoder re-ranking
6. Gemini generation (shared prompt, temperature 0)
7. Pipeline 1 and Pipeline 2
8. Failure-mode screening of 30 test queries (keyword traps, exceptions, negation)
9. Rank-shift tables and Answer 1 vs Answer 2
10. Latency profiling
11. Scaling experiment (Cross-Encoder over the full corpus)

## Notes

- On Apple Silicon, `faiss` must be imported after `torch`, and the model weights are copied into aligned memory after loading (see `realign_weights`). Without this the Cross-Encoder returns NaN scores.
