from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

@dataclass(frozen=True)
class Candidate:
    name: str
    estimator: object
    search_space: Dict[str, object] | None = None

def build_candidates(random_seed: int = 42) -> Dict[str, Candidate]:
    candidates: Dict[str, Candidate] = {}

    candidates["ridge"] = Candidate(
        name="ridge",
        estimator=Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(random_state=random_seed)),
        ]),
        search_space={
            "model__alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0],
        },
    )

    candidates["random_forest"] = Candidate(
        name="random_forest",
        estimator=RandomForestRegressor(
            random_state=random_seed,
            n_estimators=500,
        ),
        search_space={
            "max_depth": [5, 8, 12, 15, 20, None],
            "min_samples_split": [2, 5, 10, 15],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.5, 0.8, None],
        },
    )

    candidates["extra_trees"] = Candidate(
        name="extra_trees",
        estimator=ExtraTreesRegressor(
            random_state=random_seed,
            n_estimators=500,
        ),
        search_space={
            "max_depth": [15, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [2],
            "max_features": [0.5, "sqrt"],
        },
    )

    candidates["gradient_boosting"] = Candidate(
        name="gradient_boosting",
        estimator=GradientBoostingRegressor(random_state=random_seed),
        search_space={
            "n_estimators": [200, 400, 600, 800, 1000, 1500],
            "learning_rate": [0.01, 0.02, 0.05, 0.1],
            "max_depth": [2, 3, 4, 5],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_samples_leaf": [1, 2, 4, 8],
            "min_samples_split": [2, 5, 10, 20],
            "max_features": [None, "sqrt", "log2", 0.8],
            "loss": ["squared_error", "absolute_error", "huber"],
            "criterion": ["friedman_mse", "squared_error"],
            "min_weight_fraction_leaf": [0.0, 0.05],
            "max_leaf_nodes": [None, 16, 32, 64],
        },
    )

    candidates["hist_gradient_boosting"] = Candidate(
        name="hist_gradient_boosting",
        estimator=HistGradientBoostingRegressor(random_state=random_seed),
        search_space={
            "learning_rate": [0.01, 0.02, 0.05, 0.1, 0.2],
            "max_iter": [100, 200, 400, 600],
            "max_depth": [3, 5, 8, 10, None],
            "l2_regularization": [0.0, 0.1, 1.0, 10.0],
            "max_leaf_nodes": [15, 31, 63],
        },
    )

    return candidates
