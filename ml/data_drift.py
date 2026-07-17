# ml/data_drift.py
"""
Feature distribution drift detector.

The book discusses model monitoring dashboards (Grafana + MLRun) that track
both model performance drift (AUC dropping) and data drift (feature distributions
shifting from training time).

This module implements the core statistical test:
  Population Stability Index (PSI) — standard metric in financial ML.
  PSI < 0.10: No significant change (GREEN)
  PSI 0.10–0.25: Moderate shift, worth investigating (YELLOW)
  PSI > 0.25: Significant shift, likely causing model degradation (RED)

HOW TO USE:
  1. At training time, call save_reference_distribution() on X_train (the raw df).
  2. At regular intervals (daily/weekly), call compute_drift_report() with
     a sample of recent serving requests.
  3. Log or alert on features with PSI > 0.10.

HARDWARE NOTE:
  This runs entirely in Python/numpy. Fine on your machine.
  To run this on a schedule, use a GitHub Action (see .github/workflows/drift.yml)
  or a simple cron job that hits your /admin/drift endpoint.
"""

import json
import os
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

REFERENCE_PATH = "ml/models/feature_reference_dist.json"
DRIFT_REPORT_PATH = "ml/reports/drift_report.json"

# PSI thresholds (standard in financial industry)
PSI_GREEN  = 0.10
PSI_YELLOW = 0.25


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index between expected (training) and actual (serving) distributions.

    PSI = sum[(actual% - expected%) * ln(actual% / expected%)]

    Parameters:
        expected: 1D array of training values for one feature
        actual:   1D array of recent serving values for the same feature
        buckets:  number of quantile bins to use

    Returns:
        PSI value (float)
    """
    # Remove NaN
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Define bin edges from the expected (training) distribution
    quantiles = np.linspace(0, 100, buckets + 1)
    breakpoints = np.percentile(expected, quantiles)

    # Ensure unique breakpoints (needed for np.digitize)
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    # Compute bucket frequencies for each distribution
    def bucket_fractions(data, breakpoints):
        counts = np.zeros(len(breakpoints) - 1)
        for i in range(len(breakpoints) - 1):
            mask = (data >= breakpoints[i]) & (data < breakpoints[i + 1])
            counts[i] = mask.sum()
        # Add the last point to the last bucket
        counts[-1] += (data >= breakpoints[-1]).sum()
        # Avoid division by zero: replace 0 with a small number
        fracs = counts / len(data)
        fracs = np.where(fracs == 0, 1e-6, fracs)
        return fracs

    exp_fracs = bucket_fractions(expected, breakpoints)
    act_fracs = bucket_fractions(actual, breakpoints)

    psi = np.sum((act_fracs - exp_fracs) * np.log(act_fracs / exp_fracs))
    return float(psi)


def save_reference_distribution(X_train: pd.DataFrame, output_dir: str):
    """
    Save the mean and std of each numerical feature from the training set.
    This is the "expected" distribution for PSI comparison.
    Called once at the end of training.
    """
    ref = {}
    for col in X_train.select_dtypes(include=[np.number]).columns:
        values = X_train[col].dropna().tolist()
        ref[col] = values   # save raw values for PSI quantile computation

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "feature_reference_dist.json")
    with open(path, "w") as f:
        json.dump(ref, f)

    logger.info(f"Reference distribution saved: {len(ref)} features → {path}")


def compute_drift_report(recent_df: pd.DataFrame, model_dir: str) -> dict:
    """
    Compare recent serving requests to the training distribution.
    Returns a dict with PSI per feature and overall drift status.

    Parameters:
        recent_df: DataFrame of recent transactions (from MongoDB or a log file)
                   Must contain same numerical columns as training data.
        model_dir: directory containing feature_reference_dist.json

    Example usage (in a scheduled script or /admin/drift endpoint):
        from database.mongo import transaction_collection
        docs = await transaction_collection.find().sort('created_at', -1).limit(500).to_list(500)
        recent_df = pd.DataFrame(docs)
        report = compute_drift_report(recent_df, 'ml/models')
    """
    ref_path = os.path.join(model_dir, "feature_reference_dist.json")
    if not os.path.exists(ref_path):
        logger.warning("No reference distribution found. Run training first.")
        return {}

    with open(ref_path) as f:
        reference = json.load(f)

    report = {"features": {}, "summary": {}}
    high_drift_features = []

    for feature, ref_values in reference.items():
        if feature not in recent_df.columns:
            continue

        actual_values = recent_df[feature].dropna().values
        if len(actual_values) < 10:
            continue   # not enough data to be meaningful

        expected_values = np.array(ref_values)
        psi = compute_psi(expected_values, actual_values)

        if psi < PSI_GREEN:
            status = "GREEN"
        elif psi < PSI_YELLOW:
            status = "YELLOW"
            high_drift_features.append(feature)
        else:
            status = "RED"
            high_drift_features.append(feature)

        report["features"][feature] = {"psi": round(psi, 4), "status": status}

    n_red    = sum(1 for v in report["features"].values() if v["status"] == "RED")
    n_yellow = sum(1 for v in report["features"].values() if v["status"] == "YELLOW")
    n_green  = sum(1 for v in report["features"].values() if v["status"] == "GREEN")

    report["summary"] = {
        "total_features": len(report["features"]),
        "red":    n_red,
        "yellow": n_yellow,
        "green":  n_green,
        "alarm":  n_red > 0,
        "drifting_features": high_drift_features,
    }

    # Save report to disk
    os.makedirs("ml/reports", exist_ok=True)
    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    if n_red > 0:
        logger.warning(f"🚨 DRIFT ALARM: {n_red} features in RED state: "
                       f"{[f for f,v in report['features'].items() if v['status']=='RED']}")
    elif n_yellow > 0:
        logger.info(f"⚠ Drift warning: {n_yellow} features showing moderate shift.")
    else:
        logger.info(f"✓ No significant drift detected.")

    return report