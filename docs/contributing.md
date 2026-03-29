---
title: Contributing
---

# Contributing

See [CONTRIBUTING.md](https://github.com/turnert2005/artefex/blob/main/CONTRIBUTING.md) for full guidelines.

## Quick setup

```bash
git clone https://github.com/turnert2005/artefex.git
cd artefex
pip install -e ".[dev]"
pytest tests/ -v
```

## Finding work

- [Good first issues](https://github.com/turnert2005/artefex/labels/good%20first%20issue) - scoped tasks with mentorship
- [Help wanted](https://github.com/turnert2005/artefex/labels/help%20wanted) - community help appreciated
- [Feature requests](https://github.com/turnert2005/artefex/labels/enhancement) - ideas waiting for a contributor
- [Discussions](https://github.com/turnert2005/artefex/discussions) - propose your own idea

## Before submitting

```bash
pytest tests/ -v              # all tests pass
ruff check src/ tests/        # no lint errors
```
