"""Finding an expert is a search problem.

All expert access goes through one search-shaped interface: you give it a
natural-language query and it returns ranked candidate matches; you then load
the full payload for a chosen match by id. This is deliberately NOT a registry
or an enumerable store -- in the real version the space of experts can't be
listed (some are minted on demand), so you query it.

The MVP ships one implementation, `LocalExpertSearch`, whose index is backed by
the `experts/` directory and currently contains exactly one expert (Karpathy).
Growing the index -- more folders, a vector DB, on-the-fly minting -- only
changes the implementation below; nothing else in the app depends on files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Expert:
    """The full payload needed to run a chat as this expert."""

    id: str
    name: str
    tagline: str
    persona: str
    elevenlabs_voice_id: str
    transcript: str


@dataclass(frozen=True)
class ExpertMatch:
    """A lightweight search hit: enough to describe a candidate, not the corpus."""

    id: str
    name: str
    tagline: str
    persona: str


class ExpertSearch(Protocol):
    """The one seam through which experts are discovered and loaded."""

    def search(self, query: str) -> list[ExpertMatch]:
        """Return ranked candidates for a query (may be empty)."""
        ...

    def get(self, id: str) -> Expert:
        """Load the full payload for a chosen match."""
        ...


class LocalExpertSearch:
    """An `ExpertSearch` backed by the local `experts/` directory.

    Each expert is a folder with a `config.json` and a `transcripts/` directory
    of hand-supplied `.txt` files (concatenated to form the corpus). The index
    is read once at construction; `search` is intentionally a stub here -- with
    a single expert there's nothing to rank, and the *relevance* judgment for
    the MVP lives in the lobby (Gemini decides whether a query matches). As the
    index grows this is where real ranked retrieval would go.
    """

    def __init__(self, root: Path):
        self._root = root
        self._configs: dict[str, dict] = {}
        for config_path in sorted(root.glob("*/config.json")):
            config = json.loads(config_path.read_text())
            self._configs[config["id"]] = config

    @property
    def candidates(self) -> list[ExpertMatch]:
        """All indexed experts as match objects (used to prompt the lobby)."""
        return [
            ExpertMatch(
                id=c["id"],
                name=c["name"],
                tagline=c["tagline"],
                persona=c["persona"],
            )
            for c in self._configs.values()
        ]

    def search(self, query: str) -> list[ExpertMatch]:
        # MVP: a single-entry index has nothing to rank. Return every candidate
        # and let the caller (lobby + Gemini) judge relevance. Real retrieval
        # (embeddings, re-rank) slots in here without touching the interface.
        return self.candidates

    def get(self, id: str) -> Expert:
        config = self._configs.get(id)
        if config is None:
            raise KeyError(f"No expert with id {id!r}")
        return Expert(
            id=config["id"],
            name=config["name"],
            tagline=config["tagline"],
            persona=config["persona"],
            elevenlabs_voice_id=config.get("elevenlabs_voice_id", ""),
            transcript=self._load_transcript(id),
        )

    def _load_transcript(self, id: str) -> str:
        transcripts_dir = self._root / id / "transcripts"
        if not transcripts_dir.is_dir():
            return ""
        parts = [p.read_text() for p in sorted(transcripts_dir.glob("*.txt"))]
        return "\n\n".join(part.strip() for part in parts if part.strip())
