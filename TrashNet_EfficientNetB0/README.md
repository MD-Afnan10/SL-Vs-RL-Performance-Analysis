# Train EfficientNetB0 on TrashNet (TensorFlow/Keras)

Prerequisites: Python 3.8+ and a project-local virtual environment named `.venv`.

Setup (create and activate `.venv`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Run training (from the project root where `dataset/` exists):

```bash
python train_trashnet.py
```

Outputs (written to the `output/` directory):

- `efficientnetb0_trashnet.keras` (saved model)
- `classification_report.txt`
- `confusion_matrix.png`
- `accuracy.png`

This README intentionally keeps instructions minimal: use the `.venv` environment and `requirements.txt` for dependency management.
