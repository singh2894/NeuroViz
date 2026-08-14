"""
Layer 5 – Model Zoo and dynamic selection.

Provides:
- Registries of classification and regression models with sensible defaults.
- A selector that picks candidates based on dataset shape and categorical mix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

# Boosters are optional: the registries include them only when installed.
try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


@dataclass
class ModelSpec:
    name: str
    estimator: object
    task: str  # "classification" or "regression"


def classification_registry() -> Dict[str, ModelSpec]:
    registry = {
        "log_reg": ModelSpec(
            "log_reg",
            make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)
            ),
            "classification",
        ),
        "rf": ModelSpec(
            "rf",
            RandomForestClassifier(n_estimators=200, random_state=42),
            "classification",
        ),
        "knn": ModelSpec(
            "knn",
            make_pipeline(StandardScaler(), KNeighborsClassifier()),
            "classification",
        ),
        "svm": ModelSpec(
            "svm",
            make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
            "classification",
        ),
        "nb": ModelSpec("nb", GaussianNB(), "classification"),
        # Scale inputs and allow more epochs with early stopping to avoid convergence churn.
        "mlp": ModelSpec(
            "mlp",
            make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    max_iter=1000,
                    early_stopping=True,
                    n_iter_no_change=10,
                    random_state=42,
                    learning_rate_init=0.001,
                ),
            ),
            "classification",
        ),
    }
    if HAS_XGB:
        registry["xgb"] = ModelSpec(
            "xgb",
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
            ),
            "classification",
        )
    if HAS_LGBM:
        registry["lgbm"] = ModelSpec(
            "lgbm",
            LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            ),
            "classification",
        )
    return registry


def regression_registry() -> Dict[str, ModelSpec]:
    registry = {
        "lin_reg": ModelSpec(
            "lin_reg",
            make_pipeline(StandardScaler(), LinearRegression()),
            "regression",
        ),
        "rf_reg": ModelSpec(
            "rf_reg",
            RandomForestRegressor(n_estimators=300, random_state=42),
            "regression",
        ),
        "knn_reg": ModelSpec(
            "knn_reg",
            make_pipeline(StandardScaler(), KNeighborsRegressor()),
            "regression",
        ),
        "svr": ModelSpec("svr", make_pipeline(StandardScaler(), SVR()), "regression"),
        "mlp_reg": ModelSpec(
            "mlp_reg",
            make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    max_iter=1000,
                    early_stopping=True,
                    n_iter_no_change=10,
                    random_state=42,
                    learning_rate_init=0.001,
                ),
            ),
            "regression",
        ),
    }
    if HAS_XGB:
        registry["xgb_reg"] = ModelSpec(
            "xgb_reg",
            XGBRegressor(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            ),
            "regression",
        )
    if HAS_LGBM:
        registry["lgbm_reg"] = ModelSpec(
            "lgbm_reg",
            LGBMRegressor(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            ),
            "regression",
        )
    return registry


def select_models(
    task: str, n_rows: int, n_features: int, n_categorical: int
) -> List[ModelSpec]:
    """
    Simple heuristic selector:
    - Small data: prefer simpler models.
    - Many categorical: prefer tree/boosting with handling via one-hot; skip SVM if too wide.
    - Larger data: prefer LGBM/XGB.
    """
    if task == "classification":
        reg = classification_registry()
        candidates = list(reg.keys())
    else:
        reg = regression_registry()
        candidates = list(reg.keys())

    selected: List[str] = []

    if n_rows < 1000:
        # Small data: avoid heavy boosters
        selected = [c for c in candidates if "xgb" not in c and "lgbm" not in c]
    else:
        selected = candidates.copy()

    # If feature count is very high, drop SVM
    if n_features > 200 and "svm" in selected:
        selected.remove("svm")
    if n_features > 200 and "svr" in selected:
        selected.remove("svr")
    if n_features > 500 and "knn" in selected:
        selected.remove("knn")
    if n_features > 500 and "knn_reg" in selected:
        selected.remove("knn_reg")

    # If no or few categorical, naive bayes less useful
    if task == "classification" and n_categorical == 0 and "nb" in selected:
        selected.remove("nb")

    return [reg[name] for name in selected if name in reg]
