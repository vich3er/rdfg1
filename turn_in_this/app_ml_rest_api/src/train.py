import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    TARGET,
)


def load_data(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in BINARY_FEATURES:
        df[col] = df[col].str.strip().str.lower().map({"yes": 1, "no": 0})
    return df


def build_pipeline(regressor) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("pre", pre), ("model", regressor)])


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    return {
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
    }


def main():
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "ridge": Ridge(alpha=1.0),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
        ),
    }

    results, fitted = {}, {}
    for name, reg in candidates.items():
        pipe = build_pipeline(reg).fit(X_train, y_train)
        fitted[name] = pipe
        results[name] = evaluate(pipe, X_test, y_test)
        print(f"{name:20s} {results[name]}")

    best = max(results, key=lambda n: results[n]["r2"])
    print(f"Найкраща модель: {best}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(fitted[best], MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "best_model": best,
                "test_metrics": results[best],
                "all_models": results,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "features": FEATURES,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Збережено: {MODEL_PATH}")


if __name__ == "__main__":
    main()
