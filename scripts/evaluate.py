from __future__ import annotations

import json
from pathlib import Path

from src.veridocs import VeriDocsPipeline


def main() -> None:
    pipeline = VeriDocsPipeline()
    chunk_count = pipeline.ingest_documents("data/sample_docs")

    questions = [
        "What is the purpose of this project?",
        "Which model is used in this implementation?",
        "What database engine stores the vectors?",
    ]

    results = []
    for q in questions:
        out = pipeline.ask(q)
        results.append(
            {
                "question": q,
                "answer": out.answer,
                "status": out.verification_status,
                "confidence": out.confidence,
                "citations": out.citations,
            }
        )

    report = {
        "indexed_chunks": chunk_count,
        "results": results,
    }

    report_path = Path("reports/evaluation_run.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
