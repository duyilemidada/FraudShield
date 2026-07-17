# defenses/drift_detector.py
# Tracks model AUC across retraining cycles.
# Raises an alarm if AUC drops more than TOLERANCE from the rolling mean.
 
import json
import os
import numpy as np
 
HISTORY_PATH = "ml/models/auc_history.json"
WINDOW       = 5       # rolling window size
TOLERANCE    = 0.04    # alarm if AUC drops more than 4 points from window mean
 
def record_and_check_auc(current_auc: float) -> bool:
    """
    Record current AUC and check for significant drift.
    Returns True if an alarm should be raised.
    """
    history = []
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            history = json.load(f)
 
    history.append(current_auc)
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f)
 
    if len(history) < 2:
        print(f"[drift] AUC recorded: {current_auc:.4f} (baseline cycle)")
        return False
 
    window = history[-WINDOW-1:-1]   # previous window (excluding current)
    window_mean = np.mean(window)
    drop = window_mean - current_auc
 
    print(f"[drift] AUC: {current_auc:.4f}  |  Window mean: {window_mean:.4f}  |  Drop: {drop:.4f}")
 
    if drop > TOLERANCE:
        print(f"[ALARM] AUC dropped {drop:.4f} — possible incremental poisoning!")
        return True
    return False
