"""Production knowledge base: markdown chunks + TF-IDF retrieval."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"


@dataclass(frozen=True)
class Chunk:
    id: str
    title: str
    tags: tuple[str, ...]
    text: str
    source: str

    def as_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "tags": list(self.tags),
                "text": self.text, "source": self.source}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load_chunks(directory: Path = KNOWLEDGE_DIR) -> list[Chunk]:
    """Each `## Heading` section of each markdown file is one chunk.
    An optional `tags:` line right after the heading adds retrieval keywords."""
    chunks = []
    for path in sorted(directory.glob("*.md")):
        for section in re.split(r"^## ", path.read_text(encoding="utf-8"), flags=re.M)[1:]:
            title, _, body = section.partition("\n")
            body = body.strip()
            tags: tuple[str, ...] = ()
            if body.startswith("tags:"):
                tag_line, _, body = body.partition("\n")
                tags = tuple(t.strip() for t in tag_line[5:].split(",") if t.strip())
            chunks.append(Chunk(id=_slug(title.strip()), title=title.strip(), tags=tags,
                                text=body.strip(), source=path.name))
    return chunks


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk] | None = None):
        # Imported here so the server starts quickly on a small instance
        from sklearn.feature_extraction.text import TfidfVectorizer  # pylint: disable=import-outside-toplevel

        self.chunks = chunks if chunks is not None else load_chunks()
        if not self.chunks:
            raise ValueError("Knowledge base is empty")
        self._by_id = {c.id: c for c in self.chunks}
        # Title and tags are repeated so they weigh more than body text
        docs = [f"{c.title} {c.title} {' '.join(c.tags)} {' '.join(c.tags)} {c.text}" for c in self.chunks]
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self._matrix = self._vectorizer.fit_transform(docs)

    def get(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def search(self, query: str, k: int = 3, min_score: float = 0.05) -> list[tuple[Chunk, float]]:
        q = self._vectorizer.transform([query])
        scores = (self._matrix @ q.T).toarray().ravel()
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] >= min_score]

    def retrieve_for(self, queries: list[str], per_query: int = 2, limit: int = 8) -> list[Chunk]:
        """Union of the top results for several queries, best first, de-duplicated."""
        best: dict[str, float] = {}
        for query in queries:
            for chunk, score in self.search(query, k=per_query):
                best[chunk.id] = max(best.get(chunk.id, 0.0), score)
        ranked = sorted(best, key=best.get, reverse=True)[:limit]
        return [self._by_id[i] for i in ranked]
