# Contributing to voicedecide

Thanks for your interest. voicedecide is maintained in spare time by one person, so this guide is
written to keep reviews quick. **Reviews may take a few weeks.** Bug fixes with tests are the most
likely to be merged.

## How to contribute

1. **Open an issue first.** Describe the bug or the change you want to make and wait for a reply
   before writing code. Pull requests without an agreed issue may be closed without review.
   - Exception: typo and documentation fixes under ~20 lines can go straight to a pull request.
2. **Keep the pull request small.** One fix or feature per pull request.
3. **Add a test** for every behavior change (see below).
4. **Make sure CI is green.** Pull requests can only be merged when all tests pass.

## What is likely to be accepted

| Welcome | Please don't (or open an issue first) |
|---|---|
| Bug fixes with a test | Large refactors or style-only rewrites |
| Documentation and examples | New runtime dependencies |
| Support for more audio input formats | Packs committed to this repository |
| Benchmark results with a script to reproduce them | New features outside the current scope (HTTP server, streaming) without an agreed issue |

## Sharing packs

Packs (saved questions with prototype vectors) are **not** added to this repository. Publish yours
on the Hugging Face Hub and anyone can load it:

```python
Choice.from_pack("your-username/your-pack")
```

Say which data the prototypes came from and what license it has in the pack's README. Community
packs are not tested by the maintainer.

## Language policy

Only languages benchmarked by the maintainer are marked `tested`. A pull request cannot mark a
language as tested; open an issue with a reproducible benchmark (script + public data) instead.
Do not add accuracy figures to warnings, docs or tables unless they come from a run you can share.

## Development setup

```bash
git clone https://github.com/2004Snehal/voicedecide
cd voicedecide
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest            # fast unit tests, no model download
.venv/bin/pytest -m slow    # real model (~2.5 GB download on first run)
```

Unit tests use a fake encoder (`tests/conftest.py`), so they need no GPU and no model download.
Add new tests there in the same style.

## Reporting bugs

Use the bug report form. Include the voicedecide version, Python version, OS, device (CPU/GPU),
the model revision (`Decider().encoder.revision`) and a minimal code snippet.

## Inactive issues and pull requests

Issues and pull requests with no activity for 60 days are marked stale and closed 14 days later.
Comment to keep one open.

## License

By contributing you agree that your contributions are licensed under the Apache License 2.0.
