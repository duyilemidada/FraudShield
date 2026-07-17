# ml/feature_stats.py
"""
Save and load feature statistics computed from the training set.
Used for imputation at serving time: instead of filling missing/null
velocity features with 0, we fill with the training-set mean.

WHY THIS MATTERS (from the book's EnrichmentVotingEnsemble):
  The book's impute_policy={"*": "$mean"} does exactly this — it looks up
  the mean of each feature from the feature store stats and uses it to
  replace nulls. We replicate this manually.

  Example: if Redis is down and txn_count_24hr is null, filling with 0
  tells the model "this customer has never transacted before" — which may
  bias the score toward APPROVE even for a suspicious transaction.
  Filling with the mean tells the model "this is a typical customer" —
  a safer, more neutral assumption.

USAGE:
  After training: save_feature_stats(X_train_df, output_dir)
  At serving startup: stats = load_feature_stats(model_dir)
  At inference: imputed_value = stats.get(feature_name, 0)
"""

import json
import os
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

STATS_FILENAME = "feature_stats.json"


def save_feature_stats(X_train: pd.DataFrame, output_dir: str) -> dict:
    """
    Compute and save mean, median, and std for all numerical features.
    Called once at the end of training.

    Parameters:
        X_train: the raw (pre-preprocessing) training DataFrame
        output_dir: directory where model files are saved (ml/models/)

    Returns:
        dict of {feature_name: {mean, median, std, min, max}}
    """
    stats = {}
    for col in X_train.select_dtypes(include=[np.number]).columns:
        col_data = X_train[col].dropna()
        if len(col_data) == 0:
            continue
        stats[col] = {
            "mean":   float(col_data.mean()),
            "median": float(col_data.median()),
            "std":    float(col_data.std()),
            "min":    float(col_data.min()),
            "max":    float(col_data.max()),
        }

    output_path = os.path.join(output_dir, STATS_FILENAME)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Feature stats saved: {len(stats)} features → {output_path}")
    return stats


def load_feature_stats(model_dir: str) -> dict:
    """
    Load feature stats from disk. Returns empty dict if file doesn't exist.
    Called once at API startup (in lifespan).
    """
    path = os.path.join(model_dir, STATS_FILENAME)
    if not os.path.exists(path):
        logger.warning(f"feature_stats.json not found at {path}. Imputation will use 0.")
        return {}

    with open(path) as f:
        stats = json.load(f)

    logger.info(f"Feature stats loaded: {len(stats)} features.")
    return stats


def impute_with_stats(features: dict, stats: dict, strategy: str = "mean") -> dict:
    """
    Replace None/NaN values in a feature dict using training-set statistics.

    Parameters:
        features: dict of {feature_name: value} from get_velocity_features()
        stats:    dict loaded by load_feature_stats()
        strategy: "mean" or "median" — which stat to use for imputation

    Returns:
        New dict with nulls replaced.
    """
    imputed = {}
    for name, value in features.items():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            if name in stats:
                imputed[name] = stats[name][strategy]
                logger.debug(f"Imputed {name} with {strategy}={stats[name][strategy]:.4f}")
            else:
                imputed[name] = 0  # fallback if feature not in stats
        else:
            imputed[name] = value
    return imputed