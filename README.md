# AutoML-QSAR PyQt6 GUI (wired to your pipeline)

This project turns your AutoML-QSAR notebook/pipeline into a desktop GUI with:
- Project setup (output directory, ChEMBL ID, activity type)
- Feature selection (2D/3D/FP and combinations)
- Model selection (RF/XGB/LGBM/etc + stacking; optional GNN)
- Training with live log streaming
- Auto-save all publication figures/tables (same as your pipeline)
- Save best model as `best_model.pkl`
- Load model and predict pIC50 for new SMILES

## Quick start

### 1) Create environment
```bash
conda create -n qsar-gui python=3.10 -y
conda activate qsar-gui
pip install -r requirements.txt
```

### 2) Run the app
```bash
python -m qsar_gui.app
```

## Using your existing notebook (AutoML-QSAR.ipynb)

You have two options:

### Option A (recommended): Use the provided `pipeline_fixed.py`
This project already includes `qsar_gui/backend/pipeline_fixed.py` (a publishability-safe version with:
- scaffold split
- leakage-free preprocessing
- AD + y-scrambling + conformal metrics
- save best model `.pkl`)

The GUI uses this by default.

### Option B: Wire directly to your notebook
If you insist on using *your* ipynb exactly, the GUI has a "Use notebook" mode:
- It extracts code cells from the ipynb and executes them in a sandbox namespace.
- It then sets `CONFIG` values and calls `main()`.

**Note:** Notebook mode can run end-to-end, but feature/model selection depends on CONFIG hooks;
if your notebook doesn't support them, it will still run all features/models.

## Outputs (in your chosen project directory)
- `raw_data.csv`
- `2d_descriptors.csv`, `3d_descriptors.csv`, `fingerprints.npy`
- `enhanced_results.json`
- `comprehensive_results.png`
- `residual_analysis_<feature_set>.png`
- `best_model.pkl` (saved by GUI button or auto-save)
- `best_model_meta.json`

## Packaging
```bash
pyinstaller --onefile --windowed -n AutoMLQSAR qsar_gui/app.py
```
