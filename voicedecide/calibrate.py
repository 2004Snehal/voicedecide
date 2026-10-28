"""Scoring maths and calibration: softmax over cosine / tau, confidence, metrics, threshold picking."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_TAU = 0.01            # CLIP's learned temperature; fit() replaces it per question
TAU_GRID = (0.005, 0.01, 0.02, 0.03, 0.05, 0.1)
DEFAULT_THRESHOLDS = {"act": 0.80, "confirm": 0.50}
MIN_SUPPORT = 5               # clips needed above a threshold before trusting its precision


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def confidence(probs: np.ndarray) -> np.ndarray:
    """1 - normalized entropy: 1.0 when all mass is on one option, 0.0 when spread evenly."""
    k = probs.shape[-1]
    if k < 2:
        return np.ones(probs.shape[:-1])
    h = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=-1)
    return 1.0 - h / np.log(k)


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean())


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf, hit = probs.max(1), probs.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(total)


def macro_f1(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    f1s = []
    for c in range(k):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        denom = 2 * tp + fp + fn
        if np.sum(y == c) or np.sum(pred == c):
            f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def pick_tau(scores: np.ndarray, y: np.ndarray, grid=TAU_GRID) -> float:
    return float(min(grid, key=lambda t: nll(softmax(scores / t), y)))


def pick_threshold(p_top: np.ndarray, correct: np.ndarray, target_precision: float) -> float | None:
    """Lowest probability t such that predictions with p >= t are right at least target_precision of the time."""
    for t in np.unique(p_top):  # ascending
        m = p_top >= t
        if m.sum() < MIN_SUPPORT:
            break
        if correct[m].mean() >= target_precision:
            return float(t)
    return None


def actions(p_top: np.ndarray, thresholds: dict) -> np.ndarray:
    return np.where(p_top >= thresholds["act"], "act", np.where(p_top >= thresholds["confirm"], "confirm", "escalate"))


@dataclass
class FitReport:
    """Metrics on held-out clips. Produced by Choice.fit() and Choice.evaluate()."""

    n: int
    labels: list[str]
    tau: float
    thresholds: dict
    accuracy: float
    macro_f1: float
    ece: float
    nll: float
    action_share: dict
    action_accuracy: dict
    per_label_accuracy: dict
    min_similarity: float | None = None
    out_of_set_share: float = 0.0   # share of these (in-set) clips flagged out of set
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def __str__(self) -> str:
        lines = [
            f"clips: {self.n}   options: {len(self.labels)}   tau: {self.tau:g}",
            f"accuracy: {self.accuracy:.1%}   macro-F1: {self.macro_f1:.3f}   ECE: {self.ece:.3f}   NLL: {self.nll:.3f}",
            f"thresholds: act >= {self.thresholds['act']:.3f}, confirm >= {self.thresholds['confirm']:.3f}",
        ]
        for a in ("act", "confirm", "escalate"):
            acc = self.action_accuracy.get(a)
            lines.append(f"  {a:9s} {self.action_share.get(a, 0):6.1%} of clips"
                         + (f", accuracy {acc:.1%}" if acc is not None else ""))
        if self.min_similarity is not None:
            lines.append(f"  out of set {self.out_of_set_share:6.1%} of clips (min similarity {self.min_similarity:.3f})")
        weak = sorted(self.per_label_accuracy.items(), key=lambda kv: kv[1])[:5]
        lines.append("weakest options: " + ", ".join(f"{k} {v:.0%}" for k, v in weak))
        lines += [f"note: {n}" for n in self.notes]
        return "\n".join(lines)


def build_report(scores: np.ndarray, y: np.ndarray, labels: list[str], tau: float, thresholds: dict,
                 min_similarity: float | None = None, notes: list[str] | None = None) -> FitReport:
    k = len(labels)
    probs = softmax(scores / tau)
    pred = probs.argmax(1)
    p_top = probs.max(1)
    correct = pred == y
    act = actions(p_top, thresholds)
    share = {a: float((act == a).mean()) for a in ("act", "confirm", "escalate")}
    acc_by = {a: float(correct[act == a].mean()) for a in ("act", "confirm", "escalate") if (act == a).any()}
    per_label = {labels[c]: float(correct[y == c].mean()) for c in range(k) if (y == c).any()}
    oos = float((scores.max(1) < min_similarity).mean()) if min_similarity is not None else 0.0
    return FitReport(n=len(y), labels=list(labels), tau=tau, thresholds=dict(thresholds),
                     accuracy=float(correct.mean()), macro_f1=macro_f1(y, pred, k), ece=ece(probs, y),
                     nll=nll(probs, y), action_share=share, action_accuracy=acc_by,
                     per_label_accuracy=per_label, min_similarity=min_similarity, out_of_set_share=oos,
                     notes=list(notes or []))
