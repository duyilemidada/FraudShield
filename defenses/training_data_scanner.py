# defenses/training_data_scanner.py
# Run this on X_train_prep before calling train_and_compare().
# Flags samples the IsolationForest considers anomalous.
 
import numpy as np
from sklearn.ensemble import IsolationForest
 
def scan_training_data(X_prep, contamination=0.05):
    """
    Returns indices of samples flagged as anomalous.
    contamination: expected fraction of outliers in training data.
    """
    iso = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=42, n_jobs=-1)
    iso.fit(X_prep)
    preds = iso.predict(X_prep)   # -1 = anomaly, +1 = inlier
    anomaly_indices = np.where(preds == -1)[0]
    print(f"[scanner] Flagged {len(anomaly_indices)} / {len(X_prep)} training samples")
    return anomaly_indices
 
# Usage:
# suspect_idx = scan_training_data(X_train_prep, contamination=0.05)
# X_clean = np.delete(X_train_prep, suspect_idx, axis=0)
# y_clean = np.delete(np.array(y_train), suspect_idx)
