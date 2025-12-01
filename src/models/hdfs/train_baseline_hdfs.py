from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,      # <-- ajouté
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

from config import HDFS_FEATURE_DIR, HDFS_MODELS_DIR

# ---------------------
# CONFIG
# ---------------------
FEATURE_FILE = HDFS_FEATURE_DIR / "hdfs_features.csv"
HDFS_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------
# LOAD FEATURES
# ---------------------
def load_features(path: Path) -> pd.DataFrame:
    print(f"Chargement des features depuis {path} …")
    df = pd.read_csv(path)

    # Convertir Label → 0/1
    df["Label"] = df["Label"].map({"Normal": 0, "Anomaly": 1})

    print("Shape :", df.shape)
    print("Répartition des labels :")
    print(df["Label"].value_counts())
    return df

# ---------------------
# SPLIT DATA
# ---------------------
def split_data(df: pd.DataFrame):
    # colonnes CNT
    feature_cols = [c for c in df.columns if c.startswith("cnt_")]

    X = df[feature_cols].values.astype(np.float32)
    y = df["Label"].values.astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print("Tailles :")
    print("  train:", X_train.shape, " anomalies:", y_train.sum())
    print("  val  :", X_val.shape, " anomalies:", y_val.sum())
    print("  test :", X_test.shape, " anomalies:", y_test.sum())

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

# ---------------------
# LOGISTIC REGRESSION
# ---------------------
def train_logreg(X_train, y_train):
    print("\n=== Training Logistic Regression ===")
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf

# ---------------------
# RANDOM FOREST
# ---------------------
def train_random_forest(X_train, y_train):
    print("\n=== Training Random Forest ===")
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf

# ---------------------
# EVALUATION
# ---------------------
def eval_model(name, clf, X, y):
    print(f"\n--- Évaluation {name} ---")
    pred = clf.predict(X)
    print(confusion_matrix(y, pred))
    print(classification_report(y, pred, digits=4))

    # === AUC-ROC ===
    y_score = None
    if hasattr(clf, "predict_proba"):
        # probabilité de la classe positive (1)
        y_score = clf.predict_proba(X)[:, 1]
    elif hasattr(clf, "decision_function"):
        # certains modèles donnent un score continu ici
        y_score = clf.decision_function(X)

    if y_score is not None:
        auc = roc_auc_score(y, y_score)
        print(f"AUC-ROC: {auc:.4f}")
    else:
        print("AUC-ROC: non disponible (le modèle ne fournit pas de scores continus).")

# ---------------------
# MAIN
# ---------------------
def main():
    df = load_features(FEATURE_FILE)

    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = split_data(df)

    # Modèle 1 : Logistic Regression
    logreg = train_logreg(X_train, y_train)
    eval_model("Logistic Regression (val)", logreg, X_val, y_val)

    # Modèle 2 : Random Forest
    rf = train_random_forest(X_train, y_train)
    eval_model("Random Forest (val)", rf, X_val, y_val)

    # Choisir le meilleur → ici on suppose RF
    best = rf
    model_path = HDFS_MODELS_DIR / "hdfs_random_forest.joblib"

    joblib.dump(
        {"model": best, "feature_cols": feature_cols},
        model_path
    )

    print("\n=== Final test evaluation ===")
    eval_model("Random Forest (test)", best, X_test, y_test)

    print(f"Modèle sauvegardé → {model_path}")


if __name__ == "__main__":
    main()