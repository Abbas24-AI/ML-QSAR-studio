from __future__ import annotations
from typing import Optional, Callable, Dict, Any
import os, sys, traceback
from .config import QSARConfig
from .pipeline_fixed import train_full
from .notebook_adapter import run_notebook_pipeline
from .io_utils import ensure_dir

def run_pipeline(config: QSARConfig,
                 log_cb: Optional[Callable[[str], None]] = None,
                 use_notebook: bool = False,
                 notebook_path: Optional[str] = None) -> Dict[str, Any]:
    ensure_dir(config.output_dir)

    if use_notebook:
        if not notebook_path:
            raise ValueError("notebook_path is required in notebook mode.")
        updates = {
            "target_chembl_id": config.target_chembl_id,
            "output_dir": config.output_dir,
            "test_size": config.test_size,
            "val_size": config.val_size,
            "use_gpu": config.use_gpu,
            "n_trials": config.n_trials,
            "n_folds": config.n_folds,
        }
        run_notebook_pipeline(notebook_path, updates, log_cb=log_cb)
        return {"status": "ok", "mode": "notebook"}
    else:
        cfg = {
            "output_dir": config.output_dir,
            "target_chembl_id": config.target_chembl_id,
            "standard_type": config.standard_type,
            "standard_units": config.standard_units,
            "relation": config.relation,
            "test_size": config.test_size,
            "val_size": config.val_size,
            "seed": config.seed,
            "use_2d": config.use_2d,
            "use_3d": config.use_3d,
            "use_fp": config.use_fp,
            "fp_types": config.fp_types,
            "fp_nbits": config.fp_nbits,
            "enabled_models": config.enabled_models,
            "use_gpu": config.use_gpu,
            "gpu_available": True,  # pipeline_fixed uses xgb/lgbm gpu flags; will fallback internally if not
            "n_trials": config.n_trials,
            "n_folds": config.n_folds,
            "select_best_by": config.select_best_by,
        }
        return train_full(cfg, log_cb=log_cb)
