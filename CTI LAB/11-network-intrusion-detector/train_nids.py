"""
Train network intrusion detection models on NSL-KDD dataset.
"""

import json, time, warnings
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from joblib import dump, load

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
    # Combine for consistent encoding
    combined = pd.concat([train_df, test_df], ignore_index=True)
    cat_cols = ["protocol_type", "service", "flag"]
    for col in cat_cols:
        le = LabelEncoder(); combined[col] = le.fit_transform(combined[col].astype(str))
    n_train = len(train_df)
    train_df = combined.iloc[:n_train]; test_df = combined.iloc[n_train:]
    feature_cols = [c for c in train_df.columns if c not in ("label", "label_binary", "attack_category", "difficulty")]
    X_train = train_df[feature_cols].values.astype(np.float64); y_train = train_df["label_binary"].values.astype(np.int64)
    X_test = test_df[feature_cols].values.astype(np.float64); y_test = test_df["label_binary"].values.astype(np.int64)
    cats_test = test_df["attack_category"].values if "attack_category" in test_df.columns else None
    return X_train, X_test, y_train, y_test, feature_cols, cats_test, test_df

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
    models = [
        ("LogisticRegression", Pipeline([("scaler",StandardScaler()),("clf",LogisticRegression(max_iter=3000,C=1.0,random_state=RANDOM_STATE))])),
        ("RandomForest", Pipeline([("clf",RandomForestClassifier(n_estimators=200,max_depth=20,n_jobs=-1,random_state=RANDOM_STATE))])),
        ("GradientBoosting", Pipeline([("clf",GradientBoostingClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,random_state=RANDOM_STATE))])),
        ("KNN", Pipeline([("scaler",StandardScaler()),("clf",KNeighborsClassifier(n_neighbors=5,weights="distance"))])),
    ]
    try:
        from xgboost import XGBClassifier
        models.append(("XGBoost", Pipeline([("clf",XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,random_state=RANDOM_STATE,verbosity=0,eval_metric="logloss"))])))
    except: pass
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
    print(f"\n  {METRICS_JSON}")
    print(f"\n  {'Model':22s} {'F1':8s} {'ROC-AUC':8s} {'Acc':8s}")
    print(f"  {'─' * 50}")
    for r in sorted(results, key=lambda x: x.f1, reverse=True):
        print(f"  {r.name:22s} {r.f1:.4f}  {r.roc_auc:.4f}  {r.accuracy:.4f}")

if __name__ == "__main__":
    main()
