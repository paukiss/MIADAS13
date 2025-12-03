from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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
            "model__alpha": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
        },
    )

    candidates["random_forest"] = Candidate(
        name="random_forest",
        estimator=RandomForestRegressor(
            random_state=random_seed,
            n_estimators=500,
        ),
        search_space={
            "max_depth": [3, 5, 8, 12, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.5, 0.8, None],
        },
    )

    candidates["gradient_boosting"] = Candidate(
        name="gradient_boosting",
        estimator=GradientBoostingRegressor(random_state=random_seed),
        search_space={
            "n_estimators": [200, 400, 800],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 4],
            "subsample": [0.7, 0.9, 1.0],
        },
    )

    return candidates
