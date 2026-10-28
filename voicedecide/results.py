"""Result objects returned by Decider.decide()."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class Answer:
    """One question's decision for one clip."""

    choice: str | None            # top option; None when the audio is out of set
    probabilities: dict[str, float]
    confidence: float             # 1 - normalized entropy of the probabilities
    margin: float                 # top-1 minus top-2 probability
    action: str                   # "act" | "confirm" | "escalate"
    mode: str                     # "text" | "proto"
    similarity: float             # cosine of the best option (before softmax)
    out_of_set: bool = False
    type: str = "choice"

    @property
    def probability(self) -> float:
        return max(self.probabilities.values())

    def top(self, k: int = 3) -> list[tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)[:k]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DecisionResult:
    answers: dict[str, Answer]
    latency_ms: dict[str, float] = field(default_factory=dict)
    duration_s: float = 0.0
    model: str = ""
    model_revision: str | None = None
    source: str | None = None

    def __getitem__(self, question: str) -> Answer:
        return self.answers[question]

    def to_dict(self) -> dict:
        return {"model": self.model, "model_revision": self.model_revision, "source": self.source,
                "duration_s": self.duration_s, "latency_ms": self.latency_ms,
                "answers": {k: a.to_dict() for k, a in self.answers.items()}}

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kw)
