"""Knowledge base store for SME Credit policy, rules, terms and guidance.

Loads illustrative markdown knowledge chunks and provides indexed access.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class KnowledgeChunk:
    """A single identifiable knowledge chunk."""

    id: str
    category: str
    title: str
    content: str
    tags: list[str] = field(default_factory=list)
    summary_ur: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """Standard citation string."""
        return f"[{self.id}: {self.title}]"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "title": self.title,
            "content": self.content,
            "tags": self.tags,
            "summary_ur": self.summary_ur,
            "citation": self.citation,
        }


class KnowledgeStore:
    """In-memory store of loaded knowledge chunks."""

    def __init__(self, docs_dir: Path | None = None) -> None:
        self.docs_dir = docs_dir or Path(__file__).parent / "documents"
        self._chunks: dict[str, KnowledgeChunk] = {}
        self._load_documents()

    def _load_documents(self) -> None:
        """Scan docs directory and parse markdown chunk blocks."""
        if not self.docs_dir.exists():
            return

        for md_file in sorted(self.docs_dir.glob("*.md")):
            text = md_file.read_text(encoding="utf-8")
            self._parse_markdown_file(text, str(md_file.name))

    def _parse_markdown_file(self, text: str, source_file: str) -> None:
        """Extract chunks defined by ## Chunk <ID>: <Title>."""
        # Split on chunk headings: ## Chunk KB-...: Title
        pattern = re.compile(
            r"^##\s+Chunk\s+([A-Z0-9\-]+):\s*(.+)$",
            re.MULTILINE,
        )
        matches = list(pattern.finditer(text))

        for i, match in enumerate(matches):
            chunk_id = match.group(1).strip()
            title = match.group(2).strip()
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk_body = text[start_pos:end_pos].strip()

            category = "general"
            tags: list[str] = []
            summary_ur = ""

            # Parse metadata bullet points at the top of chunk body
            cat_match = re.search(r"-\s+\*\*Category:\*\*\s*(.+)", chunk_body)
            if cat_match:
                category = cat_match.group(1).strip()

            tags_match = re.search(r"-\s+\*\*Tags:\*\*\s*(.+)", chunk_body)
            if tags_match:
                tags = [t.strip().lower() for t in tags_match.group(1).split(",") if t.strip()]

            ur_match = re.search(r"\*Urdu Summary:\*\s*(.+)", chunk_body)
            if ur_match:
                summary_ur = ur_match.group(1).strip()

            # Clean content body (keep markdown text)
            self._chunks[chunk_id] = KnowledgeChunk(
                id=chunk_id,
                category=category,
                title=title,
                content=chunk_body,
                tags=tags,
                summary_ur=summary_ur,
                metadata={"source_file": source_file},
            )

    def get_chunk(self, chunk_id: str) -> KnowledgeChunk | None:
        """Get chunk by its ID."""
        return self._chunks.get(chunk_id)

    def list_chunks(self, category: str | None = None) -> list[KnowledgeChunk]:
        """List all chunks or filter by category."""
        if category:
            return [c for c in self._chunks.values() if c.category.lower() == category.lower()]
        return list(self._chunks.values())

    def get_all_ids(self) -> list[str]:
        return list(self._chunks.keys())


# Singleton instance
_store_instance: KnowledgeStore | None = None


def get_knowledge_store() -> KnowledgeStore:
    """Return singleton knowledge store."""
    global _store_instance
    if _store_instance is None:
        _store_instance = KnowledgeStore()
    return _store_instance
