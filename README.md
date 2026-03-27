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

## Demo Flow (2-3 Minutes)
1. Start app:

```powershell
streamlit run app.py
```

2. In sidebar, keep path as `data/sample_docs` and click **Ingest Documents**.
3. Open **Q&A** tab and ask:
	- `What is VeriDocs and what problem does it solve?`
4. Point out:
	- Answer text
	- Verification status and confidence
	- Citations
	- Evidence quotes
5. Ask an unsupported question:
	- `What is the monthly cloud hosting cost?`
6. Show refusal behavior (**Not Supported**).
7. Open **Study Notes** tab and ask topic:
	- `RAG architecture`

## Test Questions

### Supported
- What is VeriDocs and what problem does it solve?
- Which Hugging Face model is used?
- What document formats can this system ingest?
- How does verification status work in this system?

### Unsupported
- What is the monthly cloud hosting cost?
- Which SQL database stores vectors in this project?

## Screenshots
Add screenshots from your run in this section before client handoff.

- Home + Q&A result: `docs/screenshots/qa-result.png`
- Unsupported question result: `docs/screenshots/not-supported.png`
- Study notes result: `docs/screenshots/study-notes.png`

## Client Handoff Checklist
- Repository pushed and accessible
- App runs with `streamlit run app.py`
- Q&A supports citations and verification status
- Unsupported questions trigger refusal
- Study Notes tab works
- `scripts/evaluate.py` generates `reports/evaluation_run.json`
