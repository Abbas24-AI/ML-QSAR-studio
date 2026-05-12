from __future__ import annotations

import os
import json
import warnings
import platform
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import datamol as dm
from tqdm import tqdm
from chembl_webresource_client.new_client import new_client

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from rdkit.Chem.Scaffolds import MurckoScaffold
from molfeat.calc import FPCalculator
from molfeat.trans import MoleculeTransformer

try:
    # MAPIE <=0.8.x
    from mapie.regression import MapieRegressor
except Exception:
    MapieRegressor = None

SEED_DEFAULT = 42


def _safe_n_jobs(prefer_parallel: bool = True) -> int:
    """Cross-platform safe parallelism.

    macOS + GUI + joblib nested estimators can crash or hang with n_jobs=-1.
    Use conservative default for Darwin/Windows GUI workflows.
    """
    system = platform.system().lower()
    if system in {"darwin", "windows"}:
        return 1
    return -1 if prefer_parallel else 1


def _log(log_cb, msg: str) -> None:
    if log_cb:
        try:
            log_cb(str(msg))
        except Exception:
            print(msg)
    else:
        print(msg)


def _sanitize_xy(X: Any, y: Any, min_rows: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Remove rows with NaN/inf in X or y and return numeric arrays."""
    X_df = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan)
    y_ser = pd.Series(y).replace([np.inf, -np.inf], np.nan)

    valid = X_df.notna().all(axis=1) & y_ser.notna()
    X_clean = X_df.loc[valid].astype(float).values
    y_clean = y_ser.loc[valid].astype(float).values

    if X_clean.shape[0] < min_rows:
        raise ValueError(
            f"Too few valid samples after NaN/inf filtering: {X_clean.shape[0]} rows. "
            f"Need at least {min_rows}."
        )
    if X_clean.shape[1] == 0:
        raise ValueError("Feature matrix has zero columns after preprocessing.")
    return X_clean, y_clean


def _safe_cv(y: np.ndarray, requested_folds: int) -> KFold:
    n = len(y)
    n_splits = max(2, min(int(requested_folds), 5, n))
    return KFold(n_splits=n_splits, shuffle=True, random_state=SEED_DEFAULT)


def retrieve_chembl_data(
    target_chembl_id: str,
    output_dir: str,
    standard_type: str = "IC50",
    relation: str = "=",
    standard_units: str = "nM",
) -> pd.DataFrame:
    activity = new_client.activity
    res = activity.filter(
        target_chembl_id=target_chembl_id,
        standard_type=standard_type,
        relation=relation,
        standard_units=standard_units,
    ).filter(standard_value__isnull=False)

    df = pd.DataFrame.from_records(res)
    if df is None or df.empty:
        raise RuntimeError(f"No activity data returned for {target_chembl_id}")

    required = ["molecule_chembl_id", "canonical_smiles", "standard_value"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"ChEMBL response missing required columns: {missing}")

    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value", "canonical_smiles"])
    df = df[required]
    df.columns = ["chembl_id", "smiles", "IC50"]
    df["IC50"] = pd.to_numeric(df["IC50"], errors="coerce")
    df = df.dropna(subset=["IC50"])
    df = df[df["IC50"] > 0]
    df["pIC50"] = -np.log10(df["IC50"].astype(float) * 1e-9)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["pIC50"])
    df = df[df["pIC50"].between(3, 12)]

    with dm.without_rdkit_log():
        df["mol"] = df["smiles"].apply(lambda x: dm.to_mol(x) if pd.notna(x) else None)
    df = df[~df["mol"].isnull()].copy()

    if df.empty:
        raise RuntimeError("No valid molecules remained after pIC50 and RDKit/datamol filtering.")

    os.makedirs(output_dir, exist_ok=True)
    df.drop(columns=["mol"], errors="ignore").to_csv(os.path.join(output_dir, "raw_data.csv"), index=False)
    return df


def calculate_2d_descriptors_from_mols(mols: List[Any], output_dir: Optional[str] = None) -> pd.DataFrame:
    from rdkit.ML.Descriptors import MoleculeDescriptors

    desc_list = [
        "MolWt", "MolLogP", "MolMR", "TPSA", "HeavyAtomCount",
        "NumHAcceptors", "NumHDonors", "NumRotatableBonds",
        "NumAromaticRings", "NumAliphaticRings", "RingCount",
        "FractionCSP3", "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3",
        "BalabanJ", "BertzCT", "Ipc",
    ]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_list)
    values = []
    for mol in tqdm(mols, desc="2D Descriptors"):
        try:
            values.append(calculator.CalcDescriptors(mol))
        except Exception:
            values.append([np.nan] * len(desc_list))

    df_2d = pd.DataFrame(values, columns=desc_list).replace([np.inf, -np.inf], np.nan)
    threshold = max(1, int(len(df_2d) * 0.8))
    df_2d = df_2d.dropna(axis=1, thresh=threshold)
    if df_2d.shape[1] == 0:
        raise RuntimeError("All 2D descriptor columns were removed due to missing values.")

    imp = SimpleImputer(strategy="median")
    df_2d[:] = imp.fit_transform(df_2d)
    if output_dir:
        df_2d.to_csv(os.path.join(output_dir, "2d_descriptors.csv"), index=False)
    return df_2d


def calculate_3d_descriptors_from_mols(
    mols: List[Any], seed: int = SEED_DEFAULT, output_dir: Optional[str] = None
) -> pd.DataFrame:
    from rdkit.Chem import AllChem, Descriptors3D

    mols_3d = []
    for mol in tqdm(mols, desc="Generating 3D"):
        try:
            m3 = dm.copy_mol(mol)
            m3 = dm.add_hs(m3)
            params = AllChem.ETKDGv3()
            params.randomSeed = int(seed)
            ok = AllChem.EmbedMolecule(m3, params)
            if ok != 0:
                mols_3d.append(None)
                continue
            try:
                AllChem.MMFFOptimizeMolecule(m3, maxIters=200)
            except Exception:
                pass
            mols_3d.append(m3)
        except Exception:
            mols_3d.append(None)

    desc_3d_list = [
        "Asphericity", "Eccentricity", "InertialShapeFactor", "NPR1", "NPR2",
        "PMI1", "PMI2", "PMI3", "RadiusOfGyration", "SpherocityIndex",
    ]
    vals = []
    for m in tqdm(mols_3d, desc="3D Descriptors"):
        if m is None:
            vals.append([np.nan] * len(desc_3d_list))
            continue
        try:
            vals.append([
                Descriptors3D.Asphericity(m),
                Descriptors3D.Eccentricity(m),
                Descriptors3D.InertialShapeFactor(m),
                Descriptors3D.NPR1(m),
                Descriptors3D.NPR2(m),
                Descriptors3D.PMI1(m),
                Descriptors3D.PMI2(m),
                Descriptors3D.PMI3(m),
                Descriptors3D.RadiusOfGyration(m),
                Descriptors3D.SpherocityIndex(m),
            ])
        except Exception:
            vals.append([np.nan] * len(desc_3d_list))

    df_3d = pd.DataFrame(vals, columns=desc_3d_list).replace([np.inf, -np.inf], np.nan)
    imp = SimpleImputer(strategy="median")
    df_3d[:] = imp.fit_transform(df_3d)
    if output_dir:
        df_3d.to_csv(os.path.join(output_dir, "3d_descriptors.csv"), index=False)
    return df_3d


def calculate_fingerprints_from_smiles(
    smiles: List[str], fp_types: List[str], nbits: int, output_dir: Optional[str] = None
) -> np.ndarray:
    fingerprint_types = {
        "ECFP4": {"name": "ecfp", "radius": 2, "nBits": nbits},
        "ECFP6": {"name": "ecfp", "radius": 3, "nBits": nbits},
        "FCFP4": {"name": "fcfp", "radius": 2, "nBits": nbits},
        "FCFP6": {"name": "fcfp", "radius": 3, "nBits": nbits},
        "MACCS": {"name": "maccs"},
        "RDKIT": {"name": "rdkit", "nBits": nbits},
    }

    all_fps = []
    for name in fp_types:
        if name not in fingerprint_types:
            warnings.warn(f"Skipping unsupported fingerprint type: {name}")
            continue
        params = fingerprint_types[name]
        calc = FPCalculator(params["name"], **{k: v for k, v in params.items() if k != "name"})
        trans = MoleculeTransformer(calc)
        fps = np.array(trans.transform(np.array(smiles, dtype=object)))
        fps = np.nan_to_num(fps, nan=0.0, posinf=0.0, neginf=0.0)

        if fps.ndim == 1:
            fps = fps.reshape(-1, 1)
        if fps.shape[1] > 1:
            sel = VarianceThreshold(threshold=0.001)
            try:
                fps = sel.fit_transform(fps)
            except ValueError:
                fps = np.zeros((len(smiles), 1))
        all_fps.append(fps)

    X_fp = np.concatenate(all_fps, axis=1) if all_fps else np.zeros((len(smiles), 0))
    if output_dir:
        np.save(os.path.join(output_dir, "fingerprints.npy"), X_fp)
    return X_fp


def build_feature_matrix(smiles: List[str], mols: List[Any], recipe: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, np.ndarray]]:
    blocks: Dict[str, np.ndarray] = {}
    info: Dict[str, Any] = {}

    if recipe.get("use_2d", False):
        df2 = calculate_2d_descriptors_from_mols(mols)
        blocks["2D"] = df2.values
        info["2d_dim"] = df2.shape[1]

    if recipe.get("use_3d", False):
        df3 = calculate_3d_descriptors_from_mols(mols, seed=recipe.get("seed", SEED_DEFAULT))
        blocks["3D"] = df3.values
        info["3d_dim"] = df3.shape[1]

    if recipe.get("use_fp", False):
        fp = calculate_fingerprints_from_smiles(smiles, recipe.get("fp_types", ["ECFP4"]), recipe.get("fp_nbits", 1024))
        blocks["FP"] = fp
        info["fp_dim"] = fp.shape[1]

    X = np.concatenate(list(blocks.values()), axis=1) if blocks else np.zeros((len(smiles), 0))
    info["total_dim"] = X.shape[1]
    return X, info, blocks


def scaffold_split_indices(smiles: List[str], test_size: float) -> Tuple[np.ndarray, np.ndarray]:
    scaffolds: Dict[str, List[int]] = {}
    for i, smi in enumerate(smiles):
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(smi, includeChirality=False)
        except Exception:
            scaf = ""
        scaffolds.setdefault(scaf, []).append(i)

    sorted_scaffolds = sorted(scaffolds.values(), key=len, reverse=True)
    target_test = max(1, int(round(len(smiles) * test_size)))
    test_idx: List[int] = []
    for group in sorted_scaffolds:
        if len(test_idx) < target_test:
            test_idx.extend(group)

    test_set = set(test_idx)
    train_idx = np.array([i for i in range(len(smiles)) if i not in test_set], dtype=int)
    test_idx = np.array(sorted(test_set), dtype=int)

    if len(train_idx) < 5 or len(test_idx) < 1:
        idx = np.arange(len(smiles))
        train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=SEED_DEFAULT)
        train_idx, test_idx = np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)
    return train_idx, test_idx


def make_pipeline(model, do_selector: bool) -> Pipeline:
    n_jobs = _safe_n_jobs()
    steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if do_selector:
        steps.append((
            "selector",
            SelectFromModel(
                ExtraTreesRegressor(n_estimators=200, random_state=SEED_DEFAULT, n_jobs=n_jobs),
                threshold="median",
            ),
        ))
    steps.append(("model", model))
    return Pipeline(steps)


def applicability_domain_mean_dist(Xtr: np.ndarray, Xte: np.ndarray, k: int = 5) -> float:
    k = min(k, len(Xtr))
    if k < 1:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k).fit(Xtr)
    dists, _ = nn.kneighbors(Xte)
    return float(np.mean(dists))


def _configure_model_for_device(model: Any, name: str, use_gpu: bool, gpu_available: bool) -> Any:
    """Safe GPU/CPU settings. Defaults to CPU if uncertain."""
    if name == "XGBoost":
        # XGBoost 2.x prefers device='cuda'; older versions use gpu_hist.
        if use_gpu and gpu_available:
            try:
                model.set_params(tree_method="hist", device="cuda")
            except Exception:
                model.set_params(tree_method="hist")
        else:
            model.set_params(tree_method="hist")
    elif name == "LightGBM":
        if use_gpu and gpu_available:
            try:
                model.set_params(device="gpu")
            except Exception:
                model.set_params(device="cpu")
        else:
            model.set_params(device="cpu")
    return model


def _conformal_metrics(pipe: Pipeline, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
    if MapieRegressor is None or len(y_train) < 10:
        return np.nan, np.nan
    try:
        mapie = MapieRegressor(estimator=clone(pipe), method="plus", cv=min(5, len(y_train)))
        mapie.fit(X_train, y_train)
        try:
            _, y_pi = mapie.predict(X_test, alpha=0.1)
        except TypeError:
            # Some MAPIE versions use different kwargs. Skip safely.
            return np.nan, np.nan

        y_pi = np.asarray(y_pi)
        if y_pi.ndim == 3:
            lower = y_pi[:, 0, 0]
            upper = y_pi[:, 1, 0]
        elif y_pi.ndim == 2 and y_pi.shape[1] >= 2:
            lower = y_pi[:, 0]
            upper = y_pi[:, 1]
        else:
            return np.nan, np.nan

        cov90 = float(np.mean((y_test >= lower) & (y_test <= upper)))
        piw = float(np.mean(upper - lower))
        return cov90, piw
    except Exception:
        return np.nan, np.nan


def train_models_for_feature_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    enabled_models: List[str],
    use_gpu: bool,
    gpu_available: bool,
    n_trials: int,
    n_folds: int,
    log_cb=None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    results: Dict[str, Any] = {}
    residuals: Dict[str, Any] = {}

    X_train, y_train = _sanitize_xy(X_train, y_train, min_rows=max(5, min(n_folds, 5)))
    X_test, y_test = _sanitize_xy(X_test, y_test, min_rows=1)

    do_selector = X_train.shape[1] > 100
    cv = _safe_cv(y_train, n_folds)
    n_jobs_cv = _safe_n_jobs(prefer_parallel=False)
    n_jobs_model = _safe_n_jobs(prefer_parallel=True)

    base_models = {
        "RandomForest": RandomForestRegressor(random_state=SEED_DEFAULT, n_jobs=n_jobs_model, n_estimators=300),
        "XGBoost": XGBRegressor(random_state=SEED_DEFAULT, n_estimators=300, max_depth=5, learning_rate=0.05, objective="reg:squarederror"),
        "LightGBM": LGBMRegressor(random_state=SEED_DEFAULT, verbose=-1, n_estimators=300, learning_rate=0.05),
        "GradientBoosting": GradientBoostingRegressor(random_state=SEED_DEFAULT),
        "HistGB": HistGradientBoostingRegressor(random_state=SEED_DEFAULT),
        "ExtraTrees": ExtraTreesRegressor(random_state=SEED_DEFAULT, n_jobs=n_jobs_model, n_estimators=300),
        "SVM": SVR(kernel="rbf"),
        "KNN": KNeighborsRegressor(),
        "MLP": MLPRegressor(random_state=SEED_DEFAULT, max_iter=1000, hidden_layer_sizes=(100, 50), early_stopping=True),
        "ElasticNet": ElasticNet(random_state=SEED_DEFAULT, max_iter=10000),
        "PLS": PLSRegression(n_components=2),
        "Ridge": Ridge(),
    }

    models_to_run = {k: v for k, v in base_models.items() if k in enabled_models}
    if not models_to_run and "Stacking" not in enabled_models:
        raise ValueError("No valid models selected. Check enabled_models in GUI/config.")

    for name, model in models_to_run.items():
        _log(log_cb, f"Training {name}...")
        model = _configure_model_for_device(model, name, use_gpu, gpu_available)
        pipe = make_pipeline(model, do_selector=do_selector)

        try:
            cv_scores = cross_val_score(
                clone(pipe),
                X_train,
                y_train,
                cv=cv,
                scoring="r2",
                n_jobs=n_jobs_cv,
                error_score="raise",
            )
            pipe.fit(X_train, y_train)
            yhat_tr = pipe.predict(X_train)
            yhat_te = pipe.predict(X_test)
        except Exception as e:
            _log(log_cb, f"Skipping {name}: {type(e).__name__}: {e}")
            results[name] = {"error": f"{type(e).__name__}: {e}"}
            continue

        pre = pipe[:-1]
        try:
            Xtr = pre.transform(X_train)
            Xte = pre.transform(X_test)
            ad_mean = applicability_domain_mean_dist(Xtr, Xte)
        except Exception:
            ad_mean = np.nan

        cov90, piw = _conformal_metrics(pipe, X_train, y_train, X_test, y_test)

        try:
            yperm = np.random.default_rng(SEED_DEFAULT).permutation(y_train)
            pipe_s = clone(pipe)
            pipe_s.fit(X_train, yperm)
            yperm_hat = pipe_s.predict(X_test)
            yscr = float(r2_score(y_test, yperm_hat))
        except Exception:
            yscr = np.nan

        results[name] = {
            "train": {
                "r2": float(r2_score(y_train, yhat_tr)),
                "rmse": float(np.sqrt(mean_squared_error(y_train, yhat_tr))),
                "mae": float(mean_absolute_error(y_train, yhat_tr)),
            },
            "test": {
                "r2": float(r2_score(y_test, yhat_te)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, yhat_te))),
                "mae": float(mean_absolute_error(y_test, yhat_te)),
            },
            "cv_mean": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
            "ad_mean_dist": float(ad_mean) if not pd.isna(ad_mean) else np.nan,
            "coverage_90pi": cov90,
            "pi_width_mean": piw,
            "yscr_r2": yscr,
        }
        residuals[name] = {
            "actual_train": y_train.tolist(),
            "predicted_train": np.asarray(yhat_tr).tolist(),
            "actual_test": y_test.tolist(),
            "predicted_test": np.asarray(yhat_te).tolist(),
            "residuals_train": np.asarray(y_train - yhat_tr).tolist(),
            "residuals_test": np.asarray(y_test - yhat_te).tolist(),
        }

    if "Stacking" in enabled_models:
        working = {k: v for k, v in results.items() if isinstance(v, dict) and "test" in v and np.isfinite(v["test"].get("r2", np.nan))}
        if len(working) >= 2:
            top = sorted(working.keys(), key=lambda k: working[k]["test"]["r2"], reverse=True)[:3]
            estimators = []
            for n in top:
                est = clone(base_models[n])
                est = _configure_model_for_device(est, n, use_gpu, gpu_available)
                estimators.append((n, est))

            try:
                shared_steps: List[Tuple[str, Any]] = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
                if do_selector:
                    shared_steps.append((
                        "selector",
                        SelectFromModel(
                            ExtraTreesRegressor(n_estimators=200, random_state=SEED_DEFAULT, n_jobs=n_jobs_model),
                            threshold="median",
                        ),
                    ))
                shared_pre = Pipeline(shared_steps)
                XtrS = shared_pre.fit_transform(X_train, y_train)
                XteS = shared_pre.transform(X_test)

                stack = StackingRegressor(
                    estimators=estimators,
                    final_estimator=Ridge(alpha=1.0),
                    cv=min(5, len(y_train)),
                    n_jobs=n_jobs_cv,
                )
                stack.fit(XtrS, y_train)
                yhat_tr = stack.predict(XtrS)
                yhat_te = stack.predict(XteS)
                results["Stacking"] = {
                    "train": {
                        "r2": float(r2_score(y_train, yhat_tr)),
                        "rmse": float(np.sqrt(mean_squared_error(y_train, yhat_tr))),
                        "mae": float(mean_absolute_error(y_train, yhat_tr)),
                    },
                    "test": {
                        "r2": float(r2_score(y_test, yhat_te)),
                        "rmse": float(np.sqrt(mean_squared_error(y_test, yhat_te))),
                        "mae": float(mean_absolute_error(y_test, yhat_te)),
                    },
                    "cv_mean": np.nan,
                    "cv_std": np.nan,
                    "ad_mean_dist": applicability_domain_mean_dist(XtrS, XteS),
                    "coverage_90pi": np.nan,
                    "pi_width_mean": np.nan,
                    "yscr_r2": np.nan,
                }
                residuals["Stacking"] = {
                    "actual_train": y_train.tolist(),
                    "predicted_train": np.asarray(yhat_tr).tolist(),
                    "actual_test": y_test.tolist(),
                    "predicted_test": np.asarray(yhat_te).tolist(),
                    "residuals_train": np.asarray(y_train - yhat_tr).tolist(),
                    "residuals_test": np.asarray(y_test - yhat_te).tolist(),
                }
            except Exception as e:
                _log(log_cb, f"Skipping Stacking: {type(e).__name__}: {e}")
                results["Stacking"] = {"error": f"{type(e).__name__}: {e}"}
        else:
            results["Stacking"] = {"error": "Need at least two successful base models for stacking."}

    return results, residuals, {"do_selector": do_selector}


def select_best_model(all_results: Dict[str, Any], criterion: str = "test_r2") -> Tuple[str, str, float]:
    best_fs, best_model, best_score = None, None, -1e9
    for fs, models in all_results.items():
        for m, met in models.items():
            if not isinstance(met, dict) or "test" not in met:
                continue
            score = met["test"].get("r2", np.nan) if criterion == "test_r2" else met.get("cv_mean", np.nan)
            if score is None or not np.isfinite(score):
                continue
            if float(score) > best_score:
                best_score = float(score)
                best_fs, best_model = fs, m
    if best_fs is None:
        raise RuntimeError("No valid trained model found. Check enhanced_results.json for model errors.")
    return str(best_fs), str(best_model), float(best_score)


def _build_feature_sets(blocks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    feature_sets: Dict[str, np.ndarray] = {}
    if "2D" in blocks:
        feature_sets["2D"] = blocks["2D"]
    if "3D" in blocks:
        feature_sets["3D"] = blocks["3D"]
    if "FP" in blocks:
        feature_sets["FP"] = blocks["FP"]
    if "2D" in blocks and "3D" in blocks:
        feature_sets["2D+3D"] = np.concatenate([blocks["2D"], blocks["3D"]], axis=1)
    if "2D" in blocks and "FP" in blocks:
        feature_sets["2D+FP"] = np.concatenate([blocks["2D"], blocks["FP"]], axis=1)
    if "3D" in blocks and "FP" in blocks:
        feature_sets["3D+FP"] = np.concatenate([blocks["3D"], blocks["FP"]], axis=1)
    if all(k in blocks for k in ["2D", "3D", "FP"]):
        feature_sets["Combined"] = np.concatenate([blocks["2D"], blocks["3D"], blocks["FP"]], axis=1)
    if not feature_sets:
        raise ValueError("No feature sets were generated. Enable at least one of use_2d, use_3d, or use_fp.")
    return feature_sets


def train_full(config: Dict[str, Any], log_cb=None) -> Dict[str, Any]:
    seed = int(config.get("seed", SEED_DEFAULT))
    np.random.seed(seed)

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    _log(log_cb, "Retrieving ChEMBL data...")
    df = retrieve_chembl_data(
        target_chembl_id=config["target_chembl_id"],
        output_dir=output_dir,
        standard_type=config.get("standard_type", "IC50"),
        relation=config.get("relation", "="),
        standard_units=config.get("standard_units", "nM"),
    )

    smiles = df["smiles"].tolist()
    mols = df["mol"].tolist()
    y = df["pIC50"].values.astype(float)

    if len(y) < 10:
        raise RuntimeError(f"Too few valid activity records for modeling: {len(y)}")

    _log(log_cb, f"Valid molecules: {len(y)}")

    train_all_idx, test_idx = scaffold_split_indices(smiles, test_size=float(config.get("test_size", 0.2)))
    y_train_all = y[train_all_idx]

    # Safe validation split. qcut stratification can fail on duplicate y values or small datasets.
    try:
        bins = pd.qcut(y_train_all, q=min(5, len(np.unique(y_train_all))), labels=False, duplicates="drop")
        stratify = bins if len(np.unique(bins)) > 1 else None
    except Exception:
        stratify = None

    val_fraction = float(config.get("val_size", 0.1)) / max(1e-9, 1 - float(config.get("test_size", 0.2)))
    val_fraction = min(max(val_fraction, 0.05), 0.4)

    tr_rel, val_rel = train_test_split(
        np.arange(len(train_all_idx)),
        test_size=val_fraction,
        random_state=seed,
        stratify=stratify,
    )
    train_idx = train_all_idx[tr_rel]
    val_idx = train_all_idx[val_rel]
    train_val_idx = np.concatenate([train_idx, val_idx])

    recipe = {
        "use_2d": bool(config.get("use_2d", True)),
        "use_3d": bool(config.get("use_3d", True)),
        "use_fp": bool(config.get("use_fp", True)),
        "fp_types": list(config.get("fp_types", ["ECFP4", "ECFP6", "FCFP4", "FCFP6", "MACCS", "RDKIT"])),
        "fp_nbits": int(config.get("fp_nbits", 1024)),
        "seed": seed,
    }

    _log(log_cb, "Building feature matrices...")
    _, feat_info, blocks = build_feature_matrix(smiles=smiles, mols=mols, recipe=recipe)

    # Save descriptor blocks only once.
    if "2D" in blocks:
        pd.DataFrame(blocks["2D"]).to_csv(os.path.join(output_dir, "2d_descriptors.csv"), index=False)
    if "3D" in blocks:
        pd.DataFrame(blocks["3D"]).to_csv(os.path.join(output_dir, "3d_descriptors.csv"), index=False)
    if "FP" in blocks:
        np.save(os.path.join(output_dir, "fingerprints.npy"), blocks["FP"])

    feature_sets = _build_feature_sets(blocks)

    enabled_models = list(config.get("enabled_models", []))
    use_gpu = bool(config.get("use_gpu", False))
    gpu_available = bool(config.get("gpu_available", False))

    all_results: Dict[str, Any] = {}
    all_residuals: Dict[str, Any] = {}
    train_meta: Dict[str, Any] = {"feature_recipe": recipe, "feature_info": feat_info}

    for fs_name, X in feature_sets.items():
        _log(log_cb, f"\nFeature set: {fs_name} | shape={X.shape}")
        Xtr = X[train_val_idx]
        Xte = X[test_idx]
        ytr = y[train_val_idx]
        yte = y[test_idx]

        res, resids, fs_meta = train_models_for_feature_set(
            Xtr,
            ytr,
            Xte,
            yte,
            enabled_models=enabled_models,
            use_gpu=use_gpu,
            gpu_available=gpu_available,
            n_trials=int(config.get("n_trials", 30)),
            n_folds=int(config.get("n_folds", 5)),
            log_cb=log_cb,
        )
        all_results[fs_name] = res
        all_residuals[fs_name] = resids
        train_meta[fs_name] = fs_meta

    with open(os.path.join(output_dir, "enhanced_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    best_fs, best_model, best_score = select_best_model(all_results, criterion=config.get("select_best_by", "test_r2"))
    _log(log_cb, f"Best model: {best_model} on {best_fs} | R2={best_score:.4f}")

    # Save the best non-stacking model as deployable sklearn pipeline.
    if best_model == "Stacking":
        candidates = [
            (fs, m, met["test"]["r2"])
            for fs, mods in all_results.items()
            for m, met in mods.items()
            if isinstance(met, dict) and "test" in met and m != "Stacking" and np.isfinite(met["test"].get("r2", np.nan))
        ]
        candidates.sort(key=lambda x: x[2], reverse=True)
        if candidates:
            best_fs, best_model, best_score = candidates[0]
            _log(log_cb, f"Stacking selected, but saving deployable base model instead: {best_model} on {best_fs}")

    X_best = feature_sets[best_fs]
    Xtr = X_best[train_val_idx]
    ytr = y[train_val_idx]

    n_jobs_model = _safe_n_jobs(prefer_parallel=True)
    model_map = {
        "RandomForest": RandomForestRegressor(random_state=seed, n_jobs=n_jobs_model, n_estimators=300),
        "XGBoost": XGBRegressor(random_state=seed, tree_method="hist", objective="reg:squarederror", n_estimators=300),
        "LightGBM": LGBMRegressor(random_state=seed, verbose=-1, device="cpu", n_estimators=300),
        "GradientBoosting": GradientBoostingRegressor(random_state=seed),
        "HistGB": HistGradientBoostingRegressor(random_state=seed),
        "ExtraTrees": ExtraTreesRegressor(random_state=seed, n_jobs=n_jobs_model, n_estimators=300),
        "SVM": SVR(kernel="rbf"),
        "KNN": KNeighborsRegressor(),
        "MLP": MLPRegressor(random_state=seed, max_iter=1000, hidden_layer_sizes=(100, 50), early_stopping=True),
        "ElasticNet": ElasticNet(random_state=seed, max_iter=10000),
        "PLS": PLSRegression(n_components=2),
        "Ridge": Ridge(),
    }

    estimator = model_map.get(best_model, RandomForestRegressor(random_state=seed, n_jobs=n_jobs_model))
    estimator = _configure_model_for_device(estimator, best_model, use_gpu, gpu_available)
    do_selector = Xtr.shape[1] > 100
    pipe = make_pipeline(estimator, do_selector=do_selector)
    pipe.fit(Xtr, ytr)

    import joblib

    joblib.dump(pipe, os.path.join(output_dir, "best_model.pkl"))
    meta = {
        "best_feature_set": best_fs,
        "best_model": best_model,
        "best_score": best_score,
        "feature_recipe": recipe,
        "feature_info": feat_info,
        "platform": platform.platform(),
    }
    with open(os.path.join(output_dir, "best_model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(output_dir, "train_meta.json"), "w") as f:
        json.dump(train_meta, f, indent=2)

    # ── Collect best-model residuals for charts ────────────────────────────────
    X_test_best = X_best[test_idx]
    y_test_vals = y[test_idx]
    y_train_vals = y[train_val_idx]
    try:
        y_test_pred_vals = pipe.predict(X_test_best).tolist()
        y_train_pred_vals = pipe.predict(Xtr).tolist()
    except Exception:
        y_test_pred_vals = []
        y_train_pred_vals = []

    # ── Y-scrambling visualisation (fast, simplified model) ───────────────────
    yscr_visualization: List[float] = []
    try:
        n_jobs_scr = _safe_n_jobs(prefer_parallel=False)
        _scr_model_map = {
            "RandomForest": RandomForestRegressor(random_state=seed, n_jobs=n_jobs_scr, n_estimators=50),
            "XGBoost": XGBRegressor(random_state=seed, tree_method="hist", objective="reg:squarederror", n_estimators=50, verbosity=0),
            "LightGBM": LGBMRegressor(random_state=seed, verbose=-1, device="cpu", n_estimators=50),
            "GradientBoosting": GradientBoostingRegressor(random_state=seed, n_estimators=50),
            "HistGB": HistGradientBoostingRegressor(random_state=seed, max_iter=50),
            "ExtraTrees": ExtraTreesRegressor(random_state=seed, n_jobs=n_jobs_scr, n_estimators=50),
        }
        scr_est = _scr_model_map.get(best_model)
        if scr_est is None:
            scr_est = Ridge()
        scr_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", scr_est),
        ])
        rng_scr = np.random.default_rng(seed + 99)
        for _ in range(15):
            yperm = rng_scr.permutation(y_train_vals)
            ps = clone(scr_pipe)
            try:
                ps.fit(Xtr, yperm)
                yscr_visualization.append(float(r2_score(y_test_vals, ps.predict(X_test_best))))
            except Exception:
                yscr_visualization.append(float("nan"))
    except Exception:
        pass

    return {
        "results": all_results,
        "best": meta,
        "residuals": {
            "y_test": y_test_vals.tolist(),
            "y_pred": y_test_pred_vals,
            "y_train": y_train_vals.tolist(),
            "y_pred_train": y_train_pred_vals,
        },
        "yscramble_viz": yscr_visualization,
        "n_molecules": int(len(y)),
        "n_feature_sets": int(len(feature_sets)),
    }
