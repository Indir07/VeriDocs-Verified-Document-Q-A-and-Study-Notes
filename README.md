# VeriDocs (Fresh Build)

VeriDocs is a class-ready AI system that uses a lightweight Retrieval-Augmented Generation (RAG) flow with Hugging Face.

## What it does
- Ingests `.pdf`, `.docx`, `.txt`, `.md`
- Builds a TF-IDF retrieval index
- Uses `google/flan-t5-small` to answer from retrieved evidence
- Adds verification status and confidence score
- Includes citations and evidence quotes
- Provides a Streamlit demo UI

## Project Layout
- `app.py` - Streamlit interface
- `src/veridocs/` - ingestion, retrieval, generation, verification, pipeline
- `data/sample_docs/` - demo docs
- `scripts/evaluate.py` - baseline evaluation script
- `reports/` - generated outputs
- `logs/run_log.jsonl` - runtime log traces

## Quick Start (Windows)
```powershell
cd "d:\New Project\Hugging face Project"
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Run Evaluation
```powershell
python scripts/evaluate.py
```

Output:
- `reports/evaluation_run.json`
- `logs/run_log.jsonl`
