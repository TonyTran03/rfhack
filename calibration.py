# calibration.py
# Run from the project root:   python calibration.py
# Output:                      rfhack/data/results/calibration.csv

from pathlib import Path
import numpy as np
import pandas as pd
from rfhack.core.loaders import load_breast, load_diabetes, load_HIV
from rfhack.core.metrics import stratified_subsample
from rfhack.models.bootstrap import sample_bootstrap
from rfhack.models.gmm import sample_gmm
from rfhack.models.cvae import train_cvae_on_arrays, sample_cvae_dataset
from rfhack.core.adversarial_hacker import AdversarialHacker
from rfhack.core.rf_wrapper import RFWrapper
from rfhack.core.metrics import tstr_f1

# Config 
OUTDIR = Path("rfhack/data/results")
OUTDIR.mkdir(parents=True, exist_ok=True)

DATASETS = [load_HIV, load_breast, load_diabetes]
FRACTIONS = [0.2, 0.3, 0.4, 0.5]
TARGET_AUC = 0.55
CVAE_EPOCHS = 100   
SEED = 42


def _sample_cvae(X, y, n0, n1, seed=SEED):
    best = train_cvae_on_arrays(X, y, seed=seed, epochs=CVAE_EPOCHS, batch_size=32)
    return sample_cvae_dataset(best, n0, n1, seed=seed)

MODELS = [
    ("bootstrap", sample_bootstrap),
    ("gmm",       sample_gmm),
    ("cvae",      _sample_cvae),
]


# Helpers

def discriminator_auc(X_real, X_syn, feature_names):
    """
    Label real=1 / synthetic=0, combine, score with RFWrapper.
    Returns (mean_auc, min_auc, max_auc).
    Uses RFWrapper.from_combined which runs stratified repeated splits
    internally — same protocol as AdversarialHacker uses internally.
    """
    real_df  = pd.DataFrame(X_real, columns=feature_names)
    synth_df = pd.DataFrame(X_syn,  columns=feature_names)
    real_df['target']  = 1
    synth_df['target'] = 0
    combined = pd.concat([real_df, synth_df], ignore_index=True)
    avg, mn, mx = RFWrapper.from_combined(combined)
    return avg, mn, mx


def run_hack(X_small, feature_names, target_auc=TARGET_AUC):
    """
    Binary search over noise sigma to find synthetic data at target_auc.
    Adds explicit convergence flag so non-convergence (e.g. diabetes) is clear.
    """
    df     = pd.DataFrame(X_small, columns=feature_names)
    hacker = AdversarialHacker(df)
    result = hacker.hack(target_auc)

    return {
        "hack_sigma":     result.sigma,
        "hack_auc":       result.auc,
        "hack_auc_min":   result.auc_min,
        "hack_auc_max":   result.auc_max,
        "hack_iter":      result.iterations,
        "hack_converged": abs(result.auc - target_auc) < hacker.tol,
    }


def hack_on_model(X_syn, feature_names, target_auc=TARGET_AUC):
    df = pd.DataFrame(np.asarray(X_syn, dtype=np.float32), columns=feature_names)
    hacker = AdversarialHacker(df)
    result = hacker.hack(target_auc)
    return {
        "modelhack_sigma":     result.sigma,
        "modelhack_auc":  result.auc,
        "modelhack_auc_min":   result.auc_min,
        "modelhack_auc_max":   result.auc_max,
        "modelhack_iter":      result.iterations,
        "modelhack_converged": abs(result.auc - target_auc) < hacker.tol,
    }


# Main
rows = []

for load_fn in DATASETS:
    data = load_fn()
    X = data["X"]
    y = data["y"]
    feature_names = data["feature_names"]
    name = data["dataset"]
    p = X.shape[1]

    print(f"\n{'='*60}")
    print(f"{name}  (n={len(X)}, p={p})")
    print(f"  class counts: 0={int((y==0).sum())}  1={int((y==1).sum())}")

    for frac in FRACTIONS:
        n0 = max(2, int(np.floor((y == 0).sum() * frac)))
        n1 = max(2, int(np.floor((y == 1).sum() * frac)))

        X_small, y_small, _ = stratified_subsample(
            np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=int),
            n0=n0, n1=n1, seed=SEED,
        )

        print(f"\n  frac={frac}  n0={n0}  n1={n1}")

        hack = run_hack(X_small, feature_names)
        print(f"    [{'hack':<10}] sigma={hack['hack_sigma']:.4f}  "
              f"AUC={hack['hack_auc']:.3f} "
              f"({hack['hack_auc_min']:.3f}-{hack['hack_auc_max']:.3f})  "
              f"{'OK' if hack['hack_converged'] else 'NOT CONVERGED'}")

        row = {
            "dataset": name,
            "frac": frac,
            "n0": n0,
            "n1": n1,
            "n_features": p,
            "target_auc": TARGET_AUC,
            **hack,
        }

        # Registered models
        for model_name, sample_fn in MODELS:
            if model_name == "cvae":
                print(f"    [{model_name:<10}] training...", end=" ", flush=True)
            X_syn, y_syn = sample_fn(X_small, y_small, n0, n1, seed=SEED)
            avg, mn, mx = discriminator_auc(X_small, X_syn, feature_names)
            print(f" [{model_name:<10}] AUC={avg:.3f} ({mn:.3f}-{mx:.3f})", end="")

            row[f"{model_name}_auc"] = avg
            row[f"{model_name}_auc_min"] = mn
            row[f"{model_name}_auc_max"] = mx

            util = tstr_f1(X_small, y_small, X_syn, y_syn)
            row[f"{model_name}_tstr_f1"]    = util["tstr_f1"]
            row[f"{model_name}_trtr_f1"]    = util["trtr_f1"]
            row[f"{model_name}_utility_gap"] = util["utility_gap"]
            print(f"  F1 TRTR={util['trtr_f1']:.3f} TSTR={util['tstr_f1']:.3f} "
                  f"gap={util['utility_gap']:.3f}", end="")

            mh = hack_on_model(X_syn, feature_names)
            print(f"  →  model-hack sigma={mh['modelhack_sigma']:.4f} "
                  f"AUC={mh['modelhack_auc']:.3f} "
                  f"({'OK' if mh['modelhack_converged'] else 'NOT CONVERGED'})")
            for k, v in mh.items():
                row[f"{model_name}_{k}"] = v

        rows.append(row)

df_calib = pd.DataFrame(rows)
out_path = OUTDIR / "calibration.csv"
df_calib.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(df_calib.to_string(index=False))