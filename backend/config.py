from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class QSARConfig:
    # Project
    output_dir: str
    target_chembl_id: str
    standard_type: str = "IC50"         # IC50 / Ki / EC50 etc (ChEMBL standard_type)
    standard_units: str = "nM"          # nM default
    relation: str = "="

    # Splits
    test_size: float = 0.2
    val_size: float = 0.1
    seed: int = 42

    # Features
    use_2d: bool = True
    use_3d: bool = True
    use_fp: bool = True
    fp_types: List[str] = field(default_factory=lambda: ["ECFP4","ECFP6","FCFP4","FCFP6","MACCS","RDKIT"])
    fp_nbits: int = 1024

    # Models
    enabled_models: List[str] = field(default_factory=lambda: [
        "RandomForest","XGBoost","LightGBM","GradientBoosting","HistGB","ExtraTrees",
        "SVM","KNN","MLP","ElasticNet","PLS","Ridge","Stacking"
    ])
    use_gnn: bool = False  # Optional; requires torch+pyg in your env

    # Tuning / CV
    n_trials: int = 30
    n_folds: int = 5
    use_gpu: bool = True

    # Best-model selection
    select_best_by: str = "test_r2"     # test_r2 or cv_mean
