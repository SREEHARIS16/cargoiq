"""
CargoIQ - Model Definitions
------------------------------
Two models, both Gradient Boosted Trees (scikit-learn, so the project
has zero heavyweight/native-build dependencies — easy to deploy on
free-tier hosting):

1. DemandModel  -> predicts next-day cargo demand (tons) per route
2. RateModel    -> predicts the achieved rate per kg (dynamic pricing
                    signal), given demand/utilization context

Both are thin wrappers so training/inference code stays identical
regardless of which underlying sklearn estimator is used - swapping
in XGBoost/LightGBM later is a one-line change.
"""

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator


def build_demand_model() -> BaseEstimator:
    return GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )


def build_rate_model() -> BaseEstimator:
    return GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
