"""Retrieval and prompt construction for the four conditions."""

import math
from collections import Counter

from protocol import RETRIEVAL, SETTINGS
from rag import messages, retrieve, tokens


CLOSED_BOOK_PROMPT = (
    "Answer the question directly. Return exactly one release tag, without explanation. "
    "If you are unsure, return UNKNOWN."
)


def retrieve_bm25(question, documents):
    query_terms = dict.fromkeys(tokens(question))
    counts = [Counter(tokens(doc["text"])) for doc in documents]
    document_frequency = Counter(term for count in counts for term in count)
    lengths = [sum(count.values()) for count in counts]
    avg_length = sum(lengths) / len(documents)
    k1, b = SETTINGS["bm25_k1"], SETTINGS["bm25_b"]
    idf = {
        term: math.log1p((len(documents) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
        for term in query_terms
    }
    scores = []
    for doc, count, length in zip(documents, counts, lengths, strict=True):
        normalizer = k1 * (1 - b + b * length / avg_length)
        score = sum(
            idf[term] * count[term] * (k1 + 1) / (count[term] + normalizer)
            for term in query_terms if count[term]
        )
        scores.append({**doc, "score": score})
    scores.sort(key=lambda doc: (-doc["score"], doc["document_id"]))
    return scores[:SETTINGS["top_k"]]


def prepare_condition(condition, question, snapshots):
    if condition == "closed_book":
        return [], [
            {"role": "system", "content": CLOSED_BOOK_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ]
    documents = snapshots[RETRIEVAL[condition]]
    if condition == "bm25_updated":
        retrieved = retrieve_bm25(question, documents)
    else:
        retrieved = retrieve(question, documents, SETTINGS["top_k"])
    return retrieved, messages(question, retrieved)
