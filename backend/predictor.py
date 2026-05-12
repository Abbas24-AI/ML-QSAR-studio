from __future__ import annotations
import numpy as np
import pandas as pd
import datamol as dm
from typing import Tuple, Dict, Any, List, Optional
from .io_utils import load_model, load_json
from .pipeline_fixed import (
    calculate_2d_descriptors_from_mols,
    calculate_3d_descriptors_from_mols,
    calculate_fingerprints_from_smiles,
    build_feature_matrix
)

def predict_from_smiles(model_path: str, meta_path: str, smiles: List[str]) -> pd.DataFrame:
    model = load_model(model_path)
    meta = load_json(meta_path)

    # Parse mols
    with dm.without_rdkit_log():
        mols = [dm.to_mol(s) for s in smiles]
    valid_mask = [m is not None for m in mols]

    # Prepare features per recipe
    X, feat_info = build_feature_matrix(
        smiles=[s for s,v in zip(smiles, valid_mask) if v],
        mols=[m for m,v in zip(mols, valid_mask) if v],
        recipe=meta["feature_recipe"]
    )

    preds = model.predict(X)

    out = pd.DataFrame({
        "smiles": smiles,
        "valid": valid_mask
    })
    out["pred_pIC50"] = np.nan
    out.loc[out["valid"], "pred_pIC50"] = preds
    return out
