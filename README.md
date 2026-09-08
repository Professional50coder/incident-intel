# Incident Intelligence

Video anomaly detection portfolio project. See `docs/spec/2026-09-08-project-spec.md` for the
full design and phase map, and `.paul/PROJECT.md` / `.paul/STATE.md` for current status.

## Setup

```bash
python -m venv /d/venvs/incident-intel
source /d/venvs/incident-intel/Scripts/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

## Tests

```bash
pytest
```
