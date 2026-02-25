# Contributing to LLM Safety Middleware

Thank you for your interest in contributing! This guide will get you set up quickly.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

---

## Development Setup

### 1. Fork and clone

```bash
git clone https://github.com/SahilChachra/llm-safety-pipeline.git
cd LLM-Safety-Middleware
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies (including dev extras)

```bash
# Using uv (recommended — fast)
uv pip install -r requirements.txt

# Or with pip
pip install -e ".[dev]"
```

### 4. Install the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 5. Copy the environment template

```bash
cp .env.example .env
# Edit .env to point at a local LLM backend if you want to test generation
```

### 6. Install pre-commit hooks (optional but recommended)

```bash
pre-commit install
```

---

## Running Tests

```bash
# Run all 74 tests
pytest test_safety_pipeline.py -v

# Run a specific test class
pytest test_safety_pipeline.py::TestSafetyPipeline -v

# Run with coverage report
pytest --cov=llm_safety_pipeline --cov=api_server --cov-report=html
open htmlcov/index.html
```

All tests must pass before submitting a PR. Tests do **not** require a running LLM backend —
they mock all external HTTP calls via `respx`.

---

## Code Style

We use **black** + **isort** for formatting and **mypy** for type checking.

```bash
# Format
black .
isort .

# Check without modifying
black --check .
isort --check .

# Type check
mypy llm_safety_pipeline.py api_server.py
```

These checks run automatically in CI on every push.

---

## Submitting a Pull Request

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make your changes** — keep commits focused and well-described.

3. **Add or update tests** for any new behaviour.

4. **Update `CHANGELOG.md`** — add a line under the relevant section (`Added`, `Changed`, `Fixed`).

5. **Run the full test suite** and ensure it passes:
   ```bash
   pytest test_safety_pipeline.py -v
   black --check . && isort --check .
   ```

6. **Push and open a PR** against `main`.

### PR checklist

- [ ] Tests pass locally (`pytest -v`)
- [ ] New/changed behaviour is covered by tests
- [ ] Code formatted (`black`, `isort`)
- [ ] `CHANGELOG.md` updated
- [ ] Docstrings updated (if public API changed)

---

## Reporting Bugs

Open a [GitHub Issue](https://github.com/SahilChachra/LLM-Safety-Middleware/issues/new?template=bug_report.md) with:

- Python version and OS
- Steps to reproduce
- Expected vs actual behaviour
- Relevant log output (sanitise any sensitive data)

---

## Feature Requests

Open a [GitHub Issue](https://github.com/SahilChachra/LLM-Safety-Middleware/issues/new?template=feature_request.md) describing:

- The problem you're trying to solve
- Your proposed solution (if any)
- Alternatives you've considered

---

## Questions?

Open a [Discussion](https://github.com/SahilChachra/LLM-Safety-Middleware/discussions) — we're happy to help.
