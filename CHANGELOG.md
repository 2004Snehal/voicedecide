# Changelog

## 0.1.0a2

- PyPI page links: source, issues, changelog, author website.
- CONTRIBUTING.md, issue and pull request templates.

## 0.1.0a1

- First alpha on PyPI (same code as below).

## 0.1.0 (unreleased)

- `Decider`: embeds a clip once and answers every question from that one embedding; per-stage latency.
- `Choice.from_descriptions` (default, zero-shot text anchors) with optional centering via `adapt()`.
- `Choice.from_examples` (recommended, few-shot prototypes), `add_examples`, `remove_label`, `subset`.
- `fit` / `evaluate`: tau, act/confirm thresholds, out-of-set floor; accuracy, macro-F1, ECE, action shares.
- Packs: save/load locally or from the Hugging Face Hub; model revision check.
- Language policy: 6 tested languages load by default; other languages warn (no figures shown);
  unknown languages raise with a suggestion.
- CLI: `download`, `languages`, `build-pack`, `run`, `eval`.
