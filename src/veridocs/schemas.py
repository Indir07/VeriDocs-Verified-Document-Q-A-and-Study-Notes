from dataclasses import dataclass


@dataclass
class DocumentChunk:
    chunk_id: str
    document_name: str
    text: str


@dataclass
class AnswerResult:
    answer: str
    verification_status: str
    confidence: float
    citations: list[str]
    evidence_quotes: list[str]
