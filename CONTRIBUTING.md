# Contributing to RetinaGuard-QA

Thank you for your interest in contributing to RetinaGuard-QA.

## Guiding Principles
- **Reproducibility:** All models, metrics, and figures must be generated through version-locked scripts and documented configurations.
- **Patient Privacy & Ethics:** Raw medical images and patient health information (PHI) must never be committed to Git.
- **Strict Clinical Boundaries:** Never introduce diagnostic disease claims or unvalidated clinical efficacy assertions into code or documentation.

## Development Workflow
1. Fork the repository and create a feature branch (`git checkout -b feature/your-feature-name`).
2. Set up the local development environment (`pip install -e ".[dev]"`).
3. Ensure all code adheres to code quality standards (`ruff check .`, `black --check .`, `mypy src/`).
4. Ensure all automated tests pass (`pytest tests/ -v`).
5. Open a Pull Request describing your proposed changes and linking any relevant issues.
