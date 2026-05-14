"""In-memory inverted index for one field.

Memory layout (chosen for simplicity at ~10k docs, not for absolute size):
- ``term_to_postings``: term -> {doc_id: term_frequency}
- ``df``:               term -> document frequency
- ``doc_lens``:         doc_id -> length in tokens
- ``n_docs``, ``total_tokens`` accumulated as documents are added

Both ``avgdl`` and ``vocab_size`` are derived properties used by BM25.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass
class InvertedIndex:
    name: str
    term_to_postings: Dict[str, Dict[str, int]] = field(default_factory=dict)
    df: Dict[str, int] = field(default_factory=dict)
    doc_lens: Dict[str, int] = field(default_factory=dict)
    n_docs: int = 0
    total_tokens: int = 0

    @property
    def avgdl(self) -> float:
        return self.total_tokens / self.n_docs if self.n_docs > 0 else 0.0

    @property
    def vocab_size(self) -> int:
        return len(self.term_to_postings)

    def add_document(self, doc_id: str, tokens: Iterable[str]) -> None:
        token_list: List[str] = list(tokens) if not isinstance(tokens, list) else tokens

        # Always count the document (even if empty) so n_docs is consistent
        # across per-field indexes.
        self.doc_lens[doc_id] = len(token_list)
        self.n_docs += 1

        if not token_list:
            return

        tf = Counter(token_list)
        for term, count in tf.items():
            postings = self.term_to_postings.get(term)
            if postings is None:
                postings = {}
                self.term_to_postings[term] = postings
            postings[doc_id] = count
            self.df[term] = self.df.get(term, 0) + 1

        self.total_tokens += len(token_list)

    def postings(self, term: str) -> Dict[str, int]:
        return self.term_to_postings.get(term, {})

    def stats(self) -> dict:
        return {
            "name": self.name,
            "n_docs": self.n_docs,
            "vocab_size": self.vocab_size,
            "total_tokens": self.total_tokens,
            "avgdl": round(self.avgdl, 3),
        }
