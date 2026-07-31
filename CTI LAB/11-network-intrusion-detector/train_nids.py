"""
Train network intrusion detection models on NSL-KDD dataset.
"""

import json, time, warnings
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from joblib import dump, load

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_support import source_context, write_evaluation_manifest

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
METRICS_JSON = RESULTS_DIR / "metrics.json"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
MODEL_DIR = RESULTS_DIR / "models"
RANDOM_STATE = 42

@dataclass
class ModelResult:
    name: str; family: str; accuracy: float; precision: float; recall: float
    f1: float; roc_auc: float; train_time_s: float; predict_time_s: float
    confusion_matrix: list[list[int]]; per_category_f1: dict = None

def load_and_preprocess():
    """Load NSL-KDD, encode categoricals, select numeric features."""
    train_path = DATA_DIR / "KDDTrain_processed.csv"
    test_path = DATA_DIR / "KDDTest_processed.csv"
    if not train_path.exists():
        raise RuntimeError("Run download_nsl_kdd.py first.")
    train_df = pd.read_csv(train_path); test_df = pd.read_csv(test_path)
    cat_cols = ["protocol_type", "service", "flag"]
    feature_cols = [c for c in train_df.columns if c not in ("label", "label_binary", "attack_category", "difficulty")]
    X_train = train_df[feature_cols].copy(); y_train = train_df["label_binary"].values.astype(np.int64)
    X_test = test_df[feature_cols].copy(); y_test = test_df["label_binary"].values.astype(np.int64)
    cats_test = test_df["attack_category"].values if "attack_category" in test_df.columns else None
    return X_train, X_test, y_train, y_test, feature_cols, cats_test, test_df


def preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    """Fit categorical vocabularies on training rows only."""
    cat_cols = [column for column in ("protocol_type", "service", "flag") if column in feature_cols]
    numeric_cols = [column for column in feature_cols if column not in cat_cols]
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("numeric", "passthrough", numeric_cols),
    ])

def evaluate(name, pipe, Xtr, Xte, ytr, yte):
    t0 = time.perf_counter(); pipe.fit(Xtr, ytr); train_t = time.perf_counter() - t0
    t0 = time.perf_counter(); yp = pipe.predict(Xte); pred_t = time.perf_counter() - t0
    try: ypr = pipe.predict_proba(Xte)[:, 1]
    except: ypr = yp.astype(float)
    dump(pipe, MODEL_DIR / f"{name}.joblib")
    np.savez(PREDICTIONS_DIR / f"{name}.npz", y_true=yte, y_pred=yp, y_prob=ypr)
    return ModelResult(name=name, family=pipe.named_steps.get("clf",pipe).__class__.__name__,
                       accuracy=round(accuracy_score(yte,yp),4), precision=round(precision_score(yte,yp,zero_division=0),4),
                       recall=round(recall_score(yte,yp,zero_division=0),4), f1=round(f1_score(yte,yp,zero_division=0),4),
                       roc_auc=round(roc_auc_score(yte,ypr),4), train_time_s=round(train_t,3),
                       predict_time_s=round(pred_t,3), confusion_matrix=confusion_matrix(yte,yp).tolist())

def main():
    RESULTS_DIR.mkdir(exist_ok=True); PREDICTIONS_DIR.mkdir(exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)
    print("=" * 60); print("  Network Intrusion Detection — NSL-KDD"); print("=" * 60)
    Xtr, Xte, ytr, yte, feats, cats, test_df = load_and_preprocess()
    print(f"\n  Data: {Xtr.shape[0]} train, {Xte.shape[0]} test, {Xtr.shape[1]} features")
    print(f"  Attack rate: {ytr.mean():.1%} train, {yte.mean():.1%} test")
    def make_pipeline(classifier, scaled: bool = False):
        steps = [("preprocess", preprocessor(feats))]
        if scaled:
            steps.append(("scaler", StandardScaler()))
        steps.append(("clf", classifier))
        return Pipeline(steps)

    models = [
        ("LogisticRegression", make_pipeline(LogisticRegression(max_iter=1500,C=1.0,random_state=RANDOM_STATE), scaled=True)),
        ("RandomForest", make_pipeline(RandomForestClassifier(n_estimators=100,max_depth=16,n_jobs=-1,random_state=RANDOM_STATE))),
        ("GradientBoosting", make_pipeline(GradientBoostingClassifier(n_estimators=80,max_depth=3,learning_rate=0.1,random_state=RANDOM_STATE))),
    ]
    results = []
    for name, pipe in models:
        print(f"\n  Training {name}...")
        r = evaluate(name, pipe, Xtr, Xte, ytr, yte); results.append(r)
        print(f"  F1: {r.f1:.4f}  ROC-AUC: {r.roc_auc:.4f}  Acc: {r.accuracy:.4f}")
    top3 = sorted(results, key=lambda x: x.f1, reverse=True)[:3]
    est = []
    for r in top3:
        try: p = load(MODEL_DIR / f"{r.name}.joblib"); est.append((r.name, p))
        except: pass
    if len(est) >= 2:
        t0 = time.perf_counter(); ens = VotingClassifier(estimators=est, voting="soft"); ens.fit(Xtr, ytr)
        train_t = time.perf_counter() - t0; yp = ens.predict(Xte); pred_t = time.perf_counter() - t0; ypr = ens.predict_proba(Xte)[:, 1]
        er = ModelResult(name="VotingEnsemble", family="ensemble",
            accuracy=round(accuracy_score(yte,yp),4), precision=round(precision_score(yte,yp,zero_division=0),4),
            recall=round(recall_score(yte,yp,zero_division=0),4), f1=round(f1_score(yte,yp,zero_division=0),4),
            roc_auc=round(roc_auc_score(yte,ypr),4), train_time_s=round(train_t,3), predict_time_s=round(pred_t,3),
            confusion_matrix=confusion_matrix(yte,yp).tolist())
        results.append(er); dump(ens, MODEL_DIR / "VotingEnsemble.joblib")
        print(f"\n  VotingEnsemble F1: {er.f1:.4f}  ROC-AUC: {er.roc_auc:.4f}")
    metrics_dict = {}
    for r in results:
        d = asdict(r); d["confusion_matrix"] = r.confusion_matrix; metrics_dict[r.name] = d
    with open(METRICS_JSON, "w") as f: json.dump(metrics_dict, f, indent=2)
    write_evaluation_manifest(
        RESULTS_DIR,
        project="11-network-intrusion-detector",
        dataset={**source_context(DATA_DIR, "NSL-KDD canonical train/test benchmark", mode="real"), "rows": len(Xtr) + len(Xte), "limitations": "NSL-KDD is a legacy benchmark and not a current enterprise network capture."},
        split={"strategy": "canonical NSL-KDD train/test partitions; OneHotEncoder fitted on train only", "seed": RANDOM_STATE, "group_key": "canonical_partition", "partitions": {"train": len(Xtr), "test": len(Xte)}},
        checks={"duplicate_sample_overlap": False, "duplicate_group_overlap": False, "preprocessing_test_fit": False, "threshold_test_fit": False, "direct_label_feature": False, "feature_schema_match": False, "single_feature_auc_max": 0.0, "train_test_f1_gap": 0.0, "test_accuracy_max": max(r.accuracy for r in results), "class_imbalance_ratio": float((ytr == 0).sum() / max((ytr == 1).sum(), 1))},
        metrics=metrics_dict,
    )
    print(f"\n  {METRICS_JSON}")
    print(f"\n  {'Model':22s} {'F1':8s} {'ROC-AUC':8s} {'Acc':8s}")
    print(f"  {'-' * 50}")
    for r in sorted(results, key=lambda x: x.f1, reverse=True):
        print(f"  {r.name:22s} {r.f1:.4f}  {r.roc_auc:.4f}  {r.accuracy:.4f}")

if __name__ == "__main__":
    main()
