from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


META_FEATURES = [
    "brain_tumor_prob",
    "liver_rf_prob",
    "liver_xgb_prob",
    "liver_lgbm_prob",
    "health_model_prob",
    "kidney_model_prob",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a meta-model (stacking level-2 model) from base-model prediction probabilities."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="CSV with base model probabilities and target label columns.",
    )
    parser.add_argument(
        "--label-col",
        default="target",
        help="Name of target label column (binary: 0/1).",
    )
    parser.add_argument(
        "--output-model",
        default="meta_model.joblib",
        help="Output path for trained meta model artifact.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for validation split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, label_col: str) -> None:
    required = set(META_FEATURES + [label_col])
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in input CSV: "
            + ", ".join(missing)
            + f". Required columns are: {META_FEATURES + [label_col]}"
        )


def train_meta_model(
    df: pd.DataFrame,
    label_col: str,
    test_size: float,
    random_state: int,
) -> tuple[Pipeline, dict[str, float], str]:
    X = df[META_FEATURES].copy()
    y = df[label_col].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                META_FEATURES,
            )
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
    }
    report = classification_report(y_test, pred, digits=4)
    return model, metrics, report


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    validate_columns(df, args.label_col)

    model, metrics, report = train_meta_model(
        df=df,
        label_col=args.label_col,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_model)

    print("Meta-model training complete.")
    print(f"Saved model: {output_model.resolve()}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC:  {metrics['roc_auc']:.4f}")
    print("\nClassification report:\n")
    print(report)


if __name__ == "__main__":
    main()
