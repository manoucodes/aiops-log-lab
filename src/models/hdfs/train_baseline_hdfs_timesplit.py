from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
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
# TEMPORAL-LIKE SPLIT
# ---------------------
def temporal_split_data(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
):
    """
    Split "temporel" approximé :
    - on trie par BlockId (approx ordre dans le temps)
    - on prend:
        - 60% des premières lignes -> train
        - 20% suivantes            -> val
        - 20% restantes            -> test
    """

    # colonnes CNT = features
    feature_cols = [c for c in df.columns if c.startswith("cnt_")]

    # tri "temporel" approximatif
    df_sorted = df.sort_values("BlockId").reset_index(drop=True)

    n = len(df_sorted)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    n_test = n - n_train - n_val

    df_train = df_sorted.iloc[:n_train]
    df_val = df_sorted.iloc[n_train : n_train + n_val]
    df_test = df_sorted.iloc[n_train + n_val :]

    X_train = df_train[feature_cols].values.astype(np.float32)
    y_train = df_train["Label"].values.astype(int)

    X_val = df_val[feature_cols].values.astype(np.float32)
    y_val = df_val["Label"].values.astype(int)

    X_test = df_test[feature_cols].values.astype(np.float32)
    y_test = df_test["Label"].values.astype(int)

    print("Tailles (split temporel approximé) :")
    print("  train:", X_train.shape, " anomalies:", y_train.sum())
    print("  val  :", X_val.shape,   " anomalies:", y_val.sum())
    print("  test :", X_test.shape,  " anomalies:", y_test.sum())

    print("\nExemples de BlockId par split :")
    print("  train:", df_train["BlockId"].iloc[0], "→", df_train["BlockId"].iloc[-1])
    print("  val  :", df_val["BlockId"].iloc[0],   "→", df_val["BlockId"].iloc[-1])
    print("  test :", df_test["BlockId"].iloc[0],  "→", df_test["BlockId"].iloc[-1])

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
        random_state=42,
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

    # AUC-ROC (si le modèle supporte predict_proba)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
        print(f"AUC-ROC: {auc:.4f}")
    else:
        print("AUC-ROC indisponible (pas de predict_proba).")


# ---------------------
# MAIN
# ---------------------
def main():
    df = load_features(FEATURE_FILE)

    # ⚠️ On utilise maintenant le split temporel, pas train_test_split
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = temporal_split_data(
        df
    )

    # Modèle 1 : Logistic Regression
    logreg = train_logreg(X_train, y_train)
    eval_model("Logistic Regression (val)", logreg, X_val, y_val)

    # Modèle 2 : Random Forest
    rf = train_random_forest(X_train, y_train)
    eval_model("Random Forest (val)", rf, X_val, y_val)

    # Choisir le meilleur → on garde RF
    best = rf
    model_path = HDFS_MODELS_DIR / "hdfs_random_forest.joblib"

    print("\n=== Final test evaluation ===")
    eval_model("Random Forest (test)", best, X_test, y_test)

    joblib.dump(
        {"model": best, "feature_cols": feature_cols},
        model_path,
    )
    print(f"Modèle sauvegardé → {model_path}")


if __name__ == "__main__":
    main()