from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from .generation import HFGenerator
from .ingest import chunk_text, discover_document_files, read_document
from .retrieval import Retriever
from .schemas import AnswerResult, DocumentChunk
from .verification import verify_answer


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


class VeriDocsPipeline:
    def __init__(self, model_name: str = "google/flan-t5-small") -> None:
        self.model_name = model_name
        self.generator: HFGenerator | None = None
        self.retriever = Retriever()
        self.indexed = False

    def is_model_ready(self) -> bool:
        return self.generator is not None

    def ensure_model_loaded(self) -> None:
        if self.generator is None:
            self.generator = HFGenerator(model_name=self.model_name)

    def ingest_documents(self, path: str, log_path: str = "logs/run_log.jsonl") -> int:
        files = discover_document_files(path)
        chunks: list[DocumentChunk] = []

        for file in files:
            raw_text = read_document(file)
            for idx, piece in enumerate(chunk_text(raw_text), start=1):
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{file.name}-chunk-{idx}",
                        document_name=file.name,
                        text=piece,
                    )
                )

        self.retriever.build_index(chunks)
        self.indexed = len(chunks) > 0
        self._append_log(
            log_path,
            {
                "event": "ingest",
                "time": datetime.utcnow().isoformat(),
                "files": len(files),
                "chunks": len(chunks),
            },
        )
        return len(chunks)

    def ask(self, question: str, top_k: int = 4, log_path: str = "logs/run_log.jsonl") -> AnswerResult:
        if not self.indexed:
            return AnswerResult(
                answer="No index found. Please ingest documents first.",
                verification_status="Not Supported",
                confidence=0.0,
                citations=[],
                evidence_quotes=[],
            )

        hits = self.retriever.search(question, top_k=top_k)
        if not hits:
            return AnswerResult(
                answer="I cannot answer from the available evidence.",
                verification_status="Not Supported",
                confidence=0.0,
                citations=[],
                evidence_quotes=[],
            )

        evidence_chunks = [chunk for chunk, _ in hits]
        avg_score = sum(score for _, score in hits) / len(hits)
        context = "\n\n".join([f"[{c.chunk_id}] {c.text}" for c in evidence_chunks])

        prompt = (
            "You are a precise assistant. Answer using only the evidence. "
            "Do not copy chunk IDs, bracket labels, or markdown headers. "
            "Write a short, clean answer in 1-3 sentences. "
            "If evidence is insufficient, clearly say you cannot verify.\n\n"
            f"Question: {question}\n\n"
            f"Evidence:\n{context}\n\n"
            "Answer:"
        )
        self.ensure_model_loaded()
        answer = self._clean_answer(self.generator.generate(prompt))
        if len(answer.split()) < 4:
            answer = self._extractive_fallback(question, evidence_chunks)

        evidence_text = " ".join(c.text for c in evidence_chunks)
        status, confidence = verify_answer(answer, evidence_text, avg_score, question=question)

        if status == "Not Supported":
            answer = "I cannot verify a reliable answer from the provided documents."

        result = AnswerResult(
            answer=answer,
            verification_status=status,
            confidence=round(confidence, 3),
            citations=[f"{c.document_name} | {c.chunk_id}" for c in evidence_chunks],
            evidence_quotes=[c.text[:220] + ("..." if len(c.text) > 220 else "") for c in evidence_chunks],
        )

        self._append_log(
            log_path,
            {
                "event": "qa",
                "time": datetime.utcnow().isoformat(),
                "question": question,
                "status": result.verification_status,
                "confidence": result.confidence,
                "citations": result.citations,
            },
        )
        return result

    def generate_study_notes(self, topic: str, top_k: int = 6) -> AnswerResult:
        return self.ask(f"Create concise study notes about: {topic}", top_k=top_k)

    def _clean_answer(self, text: str) -> str:
        cleaned = text
        cleaned = re.sub(r"\[[^\]]*chunk-\d+\]", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"#{1,6}\s*", "", cleaned)
        cleaned = cleaned.replace("`", "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = cleaned.lstrip("#:- ").strip()
        return cleaned

    def _extractive_fallback(self, question: str, evidence_chunks: list[DocumentChunk]) -> str:
        q_tokens = self._meaningful_tokens(question)
        if not q_tokens:
            return ""

        best_sentence = ""
        best_score = 0

        for chunk in evidence_chunks:
            sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
            for sentence in sentences:
                s = sentence.strip()
                if not s:
                    continue
                s_tokens = self._meaningful_tokens(s)
                score = len(q_tokens.intersection(s_tokens))
                if score > best_score:
                    best_score = score
                    best_sentence = s

        return self._clean_answer(best_sentence) if best_score > 0 else ""

    def _meaningful_tokens(self, text: str) -> set[str]:
        tokens = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
        return {t for t in tokens if t not in STOPWORDS}

    def _append_log(self, log_path: str, payload: dict) -> None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
