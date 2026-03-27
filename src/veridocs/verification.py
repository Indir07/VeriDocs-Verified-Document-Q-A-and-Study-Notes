from __future__ import annotations

import re


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


def _tokenize(text: str, remove_stopwords: bool = False) -> set[str]:
    tokens = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
    if not remove_stopwords:
        return tokens
    return {t for t in tokens if t not in STOPWORDS}


def verify_answer(
    answer: str,
    evidence_text: str,
    retrieval_score: float,
    question: str | None = None,
) -> tuple[str, float]:
    answer_tokens = _tokenize(answer)
    evidence_tokens = _tokenize(evidence_text)
    if not answer_tokens or not evidence_tokens:
        return "Not Supported", 0.0

    overlap = len(answer_tokens.intersection(evidence_tokens)) / max(1, len(answer_tokens))
    question_relevance = 1.0
    if question:
        question_tokens = _tokenize(question, remove_stopwords=True)
        if question_tokens:
            question_relevance = len(question_tokens.intersection(evidence_tokens)) / len(question_tokens)
        else:
            question_relevance = 0.0

    confidence = max(
        0.0,
        min(1.0, (0.55 * overlap) + (0.25 * retrieval_score) + (0.20 * question_relevance)),
    )

    if confidence >= 0.62:
        return "Supported", confidence
    if confidence >= 0.38:
        return "Partially Supported", confidence
    return "Not Supported", confidence
