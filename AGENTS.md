# Repo Notes

- This repo is currently a minimal scaffold, not a working package: `main.py` exists but is empty, and there is no `pyproject.toml`, `requirements*.txt`, test config, lint config, formatter config, or CI workflow.
- Do not assume commands like `pytest`, `ruff`, `mypy`, or `mltrainer` exist here. If you add tooling or an entrypoint, add its config and document the exact command in the same change.
- The only checked-in implementation assets are the datasets under `dataset/` and the spec document `AGENT.md`.

# Source Of Truth

- Treat `AGENT.md` as a product/spec document in Chinese, not as a description of current code. It defines desired CLI behavior, supported models, and suggested architecture, but files like `cli.py`, `core/`, `ui/`, and `outputs/` do not exist yet.
- Trust the real filesystem over the spec when they conflict. Example: the spec refers to `data/`, but the actual datasets live in `dataset/`.

# Spec Targets

- If you implement the CLI described in `AGENT.md`, the intended commands are `list-models`, `train --dataset ...`, `compare --dataset ...`, and `optimize --dataset ...`.
- The intended model set is dataset-specific, not generic:
  - `heart`: `LR`, `DT`, `SVM`, `RF`, `KNN`
  - `winequality-red`: `Regression`, `RANSAC`, `Ridge`, `LASSO`, `ElasticNet`, `DTR`, `RFR`
- The spec's ranking metrics are also dataset-specific: rank `heart` models by `F1-score`, and rank `winequality-red` models by `RMSE`.
- The spec's optimization step is only for the recommended final model on each dataset, not every model: `Random Forest Classifier` for `heart`, `Random Forest Regressor` for `winequality-red`.
- `AGENT.md` also asks agents to report progress in Traditional Chinese and ask the user instead of guessing when blocked.

# Environment

- Use the repo-local interpreter at `.venv\Scripts\python.exe` on Windows. `.vscode/settings.json` is configured for a venv + pip workflow.
- The current venv does not have common ML/CLI packages installed: `scikit-learn`, `pandas`, `rich`, `colorama`, `typer`, and `click` were all absent when checked with `pip show`.

# Data Files

- `dataset/heart.csv` is comma-delimited and uses `target` as the label column. This is the classification dataset described in `AGENT.md`.
- `dataset/winequality-red.csv` is semicolon-delimited, not comma-delimited, and uses `quality` as the label column. This is the regression dataset described in `AGENT.md`.
- If you implement loaders, handle the two datasets explicitly instead of assuming one CSV format.

# Verification

- There is no existing automated verification path. Today the only directly verifiable runtime command is `.venv\Scripts\python.exe main.py`, and it currently does nothing because `main.py` is empty.
- When adding functionality, prefer a focused run command against the exact entrypoint you introduce rather than inventing a broader project workflow.
