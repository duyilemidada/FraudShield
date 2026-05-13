# ml/train_fraud_model.py
# End-to-End ML pipeline for FraudShield, following Géron's Chapter 4.
# Loads transactions from MongoDB, engineers features, trains multiple models,
# compares them, diagnoses overfitting via learning curves, and saves the best
# model + preprocessor + decision thresholds for use in the live API.

import pandas as pd
import numpy as np
import asyncio
import logging
import os
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    accuracy_score
)

import joblib
import database.mongo as mongo_module   # the same Motor client used by your API


# ── LOGGER ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═════════════════════════════════════════════════════════
def load_data():
    """Fetch all transactions from MongoDB and return a DataFrame."""
    logger.info("Loading transactions from MongoDB...")
    async def fetch():
        data = []
        async for doc in mongo_module.transaction_collection.find():
            data.append(doc)
        return data
    data = asyncio.run(fetch())
    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df)} transactions")
    return df 

# ═════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING (Chapter 4)
# ═════════════════════════════════════════════════════════
def engineer_features(df):
    """
    Add new features that better represent underlying patterns.
    - log_amount = log(1 + amount): compresses the wide range of transaction
      amounts so that linear models (LogisticRegression) can learn a linear
      relationship with fraud. Instead of 1000 vs 500000, we see 6.9 vs 13.1.
    """
    df = df.copy()
    # log1p = log(1 + x), safe for zero
    df['log_amount'] = np.log1p(df['amount'])
    return df


# ═════════════════════════════════════════════════════════
# 3. SEPARATE FEATURES AND LABEL
# ═════════════════════════════════════════════════════════
def split_features_labels(df):
    """
    X = all columns except the target ('is_fraud') and metadata.
    y = 'is_fraud' converted to 0/1.
    """
    drop_cols = [
        '_id', 'transaction_id', 'is_fraud', 'created_at',
        'customer_email', 'customer_phone', 'customer_ip',
        'device_fingerprint', 'merchant_id', 'fraud_score', 'decision'
    ]
    # Only drop columns that actually exist in the DataFrame
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['is_fraud'].astype(int)
    return X, y


# ═════════════════════════════════════════════════════════
# 4. BUILD PREPROCESSING PIPELINE
# ═════════════════════════════════════════════════════════
def build_preprocessing_pipeline(X):
    """
    Numerical features: impute missing values with median, then scale to
    mean 0 and variance 1 (required for LogisticRegression).
    Categorical features: one-hot encode (create a binary column per category).
    """
    numerical_cols   = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    logger.info(f"Numerical features:   {numerical_cols}")
    logger.info(f"Categorical features: {categorical_cols}")

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),      # mean=0, std=1
    ])
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline,   numerical_cols),
        ("cat", cat_pipeline,   categorical_cols),
    ])
    return full_pipeline


# ═════════════════════════════════════════════════════════
# 5. EVALUATE A SINGLE MODEL (confusion matrix + metrics)
# ═════════════════════════════════════════════════════════
def evaluate_model(name, y_true, y_pred, y_proba):
    """
    Print confusion matrix, precision, recall, F1, and ROC-AUC.
    Returns a dict with name, auc, f1 so we can compare models.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    auc = roc_auc_score(y_true, y_proba)
    f1  = f1_score(y_true, y_pred)

    logger.info(f"=== {name} ===")
    logger.info(f"  True Positives (fraud caught): {tp}")
    logger.info(f"  True Negatives (legit ok):    {tn}")
    logger.info(f"  False Positives (false alarm):{fp}")
    logger.info(f"  False Negatives (missed fraud):{fn}")
    logger.info(f"  Precision: {precision_score(y_true, y_pred):.2%}")
    logger.info(f"  Recall:    {recall_score(y_true, y_pred):.2%}")
    logger.info(f"  F1 Score:  {f1:.2%}")
    logger.info(f"  ROC-AUC:   {auc:.3f}")
    return {'name': name, 'auc': auc, 'f1': f1}


# ═════════════════════════════════════════════════════════
# 6. LEARNING CURVES (diagnose overfitting/underfitting)
# ═════════════════════════════════════════════════════════
def plot_learning_curves(model, X_train, y_train, X_val, y_val):
    """
    Train the model on increasingly larger subsets (50, 100, ... 400) and
    record training F1 and validation F1.
    - If validation F1 keeps rising → more data may help.
    - If validation F1 plateaued → model capacity is saturated.
    - Large gap between train & val → overfitting.
    """
    sizes_abs = [50, 100, 150, 200, 250, 300, 350, 400]
    train_f1s = []
    val_f1s   = []

    for size in sizes_abs:
        if size > len(X_train):
            break
        m = clone(model)            # fresh untrained copy
        m.fit(X_train[:size], y_train[:size])

        train_f1 = f1_score(y_train[:size], m.predict(X_train[:size]))
        val_f1   = f1_score(y_val, m.predict(X_val))
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

    logger.info("=== LEARNING CURVES ===")
    logger.info(f"  {'Size':>5}  {'Train F1':>8}  {'Val F1':>8}  {'Gap':>8}")
    for size, tr, va in zip(sizes_abs, train_f1s, val_f1s):
        gap = tr - va
        logger.info(f"  {size:5d}  {tr:8.3f}  {va:8.3f}  {gap:8.3f}")

    # Diagnosis
    final_val = val_f1s[-1]
    final_gap = train_f1s[-1] - final_val
    if final_gap > 0.20:
        logger.warning("⚠️  Overfitting detected – large gap between train and val.")
        logger.warning("   → Try more regularization, simpler model, or collect more data.")
    elif final_val < 0.60:
        logger.warning("⚠️  Underfitting – model is too simple or needs better features.")
        logger.warning("   → Add more features, use a more powerful model, or improve data quality.")
    else:
        logger.info("✅ Model generalizes well.")


# ═════════════════════════════════════════════════════════
# 7. FIND OPTIMAL DECISION THRESHOLDS (business logic)
# ═════════════════════════════════════════════════════════
def find_optimal_thresholds(y_true, y_proba,
                            target_precision=0.90,
                            target_recall=0.80):
    """
    Determine the probability thresholds for two‑tier decision:
    - BLOCK: auto-reject when we are at least `target_precision` sure it's fraud.
    - REVIEW: send to human if we catch at least `target_recall` of actual fraud.

    Uses precision-recall curve which returns values sorted by threshold (low→high).
    We look for the FIRST (lowest) threshold that meets each target.
    If the target cannot be met, fallback:
      BLOCK  → use the highest precision the model can achieve
               (choosing the MOST conservative threshold, i.e., highest).
      REVIEW → use the highest recall the model can achieve
               (choosing the LEAST conservative threshold, i.e., lowest).
    """
    precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_proba)

    # Helper: find the smallest threshold where condition is True
    def threshold_for_condition(condition):
        idx = np.argmax(condition)          # first True (since True=1, False=0)
        if idx < len(thresholds_pr) and condition[idx]:
            return float(thresholds_pr[idx])
        return None

    # ── Block threshold ────────────────────────────
    block_thresh = threshold_for_condition(precisions[:-1] >= target_precision)
    if block_thresh is None:
        # Fallback: best precision at the MOST CONSERVATIVE threshold (highest).
        # We take the maximum precision and then find the LAST (highest) threshold
        # that gives that precision, to avoid being too aggressive.
        max_prec = np.max(precisions[:-1])
        # Indices where precision equals max_prec, and pick the last one
        best_idx = np.where(precisions[:-1] == max_prec)[0][-1]
        block_thresh = float(thresholds_pr[best_idx])
        logger.warning(
            f"Could not reach {target_precision:.0%} precision. "
            f"Using best achievable precision: {max_prec:.2%} "
            f"at threshold {block_thresh:.4f}"
        )

    # ── Review threshold ──────────────────────────
    review_thresh = threshold_for_condition(recalls[:-1] >= target_recall)
    if review_thresh is None:
        # Fallback: best recall at the LEAST conservative threshold (lowest).
        max_rec = np.max(recalls[:-1])
        # First index where recall equals max_rec (lowest threshold)
        best_idx = np.argmax(recalls[:-1])   # first occurrence
        review_thresh = float(thresholds_pr[best_idx])
        logger.warning(
            f"Could not reach {target_recall:.0%} recall. "
            f"Using best achievable recall: {max_rec:.2%} "
            f"at threshold {review_thresh:.4f}"
        )

    # ── Safety: review must be lower than block ──
    if review_thresh >= block_thresh:
        # Force a reasonable gap
        review_thresh = block_thresh * 0.5
        logger.info(
            f"Adjusted review threshold to {review_thresh:.4f} "
            f"(half of block) to maintain decision hierarchy."
        )

    logger.info(f"Block  threshold: {block_thresh:.4f} "
                f"(precision ≈ {target_precision:.0%})")
    logger.info(f"Review threshold: {review_thresh:.4f} "
                f"(recall ≈ {target_recall:.0%})")
    return {
        "BLOCK_THRESHOLD": block_thresh,
        "REVIEW_THRESHOLD": review_thresh
    }


# ═════════════════════════════════════════════════════════
# 8. TRAIN MULTIPLE MODELS AND PICK THE BEST
# ═════════════════════════════════════════════════════════
def train_and_compare(X_train, y_train, X_val, y_val, feature_names=None):
    """
    Trains 5 models and returns the best by ROC-AUC:
      A) LogisticRegression Ridge
      B) LogisticRegression Lasso
      C) Decision Tree  (new, interpretable white box)
      D) SVM with RBF kernel
      E) RandomForest
      F) Voting Classifier (soft voting of the above five)
    """
    results = []

    # A) Logistic Regression with L2 (Ridge) – prevents large weights
    logger.info("Training Logistic Regression (Ridge)...")
    lr = LogisticRegression(
        C=1.0,              # inverse of regularization strength (lower = stronger)
        penalty='l2',       # Ridge
        solver='lbfgs',     # efficient for small/medium datasets
        max_iter=1000,
        class_weight='balanced',  # handle class imbalance
        random_state=42
    )
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_val)
    lr_proba = lr.predict_proba(X_val)[:, 1]
    results.append(evaluate_model("LogisticRegression (Ridge)", y_val, lr_pred, lr_proba))

    # B) Logistic Regression with L1 (Lasso) – can zero out useless features
    logger.info("Training Logistic Regression (Lasso)...")
    lr_l1 = LogisticRegression(
        C=0.1,
        penalty='l1',
        solver='liblinear',   # supports L1
        class_weight='balanced',
        random_state=42
    )
    lr_l1.fit(X_train, y_train)
    zeroed = np.sum(lr_l1.coef_[0] == 0)
    logger.info(f"  Lasso zeroed out {zeroed} features (ignored by the model).")
    l1_pred = lr_l1.predict(X_val)
    l1_proba = lr_l1.predict_proba(X_val)[:, 1]
    results.append(evaluate_model("LogisticRegression (Lasso)", y_val, l1_pred, l1_proba))

    # ── C: Decision Tree  ───────────────────────────
    logger.info('Training DecisionTree (interpretable)...')
    #
    # max_depth=5: shallow enough to be readable and stable.
    # min_samples_leaf=5: no leaf with fewer than 5 transactions.
    # This prevents the tree from carving out tiny fraud pockets
    # based on 1-2 examples — a classic overfit pattern.
    #

    dt = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=5,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42
    )
    dt.fit(X_train, y_train)

    dt_proba = dt.predict_proba(X_val) [:, 1]
    results.append(evaluate_model('DecisionTree', y_val, dt.predict(X_val), dt_proba))

    # Print tree rules for debugging (white box explanation)
    if feature_names:
        rules = export_text(dt, feature_names=list(feature_names))
        logger.info(f'Decision Tree rules: \n{rules}')

    # ── D: SVM with RBF kernel (Chapter 5) ───────────────────────────
    logger.info('Training SVM (RBF kernel)...')
    #
    # probability=True: required for predict_proba() to work.
    # Without this, SVM only outputs 0/1, not a probability.
    # It adds Platt scaling — slightly slower but necessary.
    #
    # GridSearchCV: tries all 16 combinations of C and gamma (4x4)
    # and picks the one with best ROC-AUC via 3-fold cross-validation.
    # n_jobs=-1 uses all CPU cores in parallel.
    #
    svm_base = SVC(
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=42
    )
    param_grid = {
        'C':     [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1]
    }
    svm_grid = GridSearchCV(
        svm_base, param_grid,
        cv=3, scoring='roc_auc', n_jobs=-1
    )
    svm_grid.fit(X_train, y_train)
    best_svm = svm_grid.best_estimator_
    logger.info(f'  Best SVM params: {svm_grid.best_params_}')

    svm_proba = best_svm.predict_proba(X_val)[:, 1]
    results.append(evaluate_model('SVM_RBF', y_val, best_svm.predict(X_val), svm_proba))


    # E) Random Forest (ensemble baseline)
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        oob_score=True
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_val)
    rf_proba = rf.predict_proba(X_val)[:, 1]
    results.append(evaluate_model("RandomForest", y_val, rf_pred, rf_proba))

    #F) Voting Classifier(ensemble of the 5 base models)
    logger.info('Training Voting Classifier (soft voting)')
    # Create an ensemble using the already trained models.
    # The VotingClassifier will fit the whole ensemble using the same training data,
    # but since the estimators are already fitted, they retain their learned weights.

    # This is fine – we just need a unified object for prediction.
    voting_clf = VotingClassifier(
        estimators=[
            ('lr_ridge', lr),
            ('lr_lasso', lr_l1),
            ('dt', dt),
            ('svm', best_svm),
            ('rf', rf)
        ],
        voting='soft',
        n_jobs=-1
    )
    
      # --- Avoid refitting: manually set fitted estimators ---
    # We already have trained models; we only need the ensemble object
    # to expose .predict_proba(). The following lines replace the normal fit()
    voting_clf.estimators_ = [est for _, est in voting_clf.estimators]
    # Also set the label encoder from any fitted model (they all know the classes)
    voting_clf.le_ = lr.classes_

    voting_pred = voting_clf.predict(X_val)
    voting_proba = voting_clf.predict_proba(X_val)[:, 1]

    results.append(evaluate_model('VotingClassifier (soft)', y_val, voting_pred, voting_proba))
    # Select the model with the highest ROC-AUC
    best = max(results, key=lambda r: r['auc'])
    logger.info(f"✅ BEST MODEL: {best['name']} (AUC={best['auc']:.3f})")

    models = {                                     
        'LogReg_Ridge':  lr,
        'LogReg_Lasso':  lr_l1,
        'DecisionTree':  dt,
        'SVM_RBF':       best_svm,
        'RandomForest':  rf,
        'VotingClassifier (soft)': voting_clf
    }
    probas = {                                    
        'LogReg_Ridge':  lr_proba,
        'LogReg_Lasso':  l1_proba,
        'DecisionTree':  dt_proba,
        'SVM_RBF':       svm_proba,
        'RandomForest':  rf_proba,
        'VotingClassifier (soft)': voting_proba
    }

    return models[best['name']], probas[best['name']]

    
    

# ═════════════════════════════════════════════════════════
# 9. MAIN PIPELINE
# ═════════════════════════════════════════════════════════
def main():
    # 1. Load
    df = load_data()
    if df.empty:
        logger.error("No data found! Run: python -m ml.seed_synthetic_data")
        return

    # 2. Feature engineering (log transform)
    df = engineer_features(df)

    # 3. Split features and labels
    X, y = split_features_labels(df)
    logger.info(f"Fraud rate: {y.mean():.1%}")

    # 4. Train/validation split (stratified to keep fraud ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # 5. Build preprocessing pipeline and fit on training data
    preprocessor = build_preprocessing_pipeline(X_train)
    X_train_prep = preprocessor.fit_transform(X_train)
    X_val_prep   = preprocessor.transform(X_val)

    try:
        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        ohe = preprocessor.named_transformers_['cat']['onehot']
        cat_names = ohe.get_feature_names_out(cat_cols).tolist()
        feature_names = num_cols + cat_names
    except Exception:
        feature_names = None

    # 6. Train all models, pick the best
    best_model, best_proba = train_and_compare(X_train_prep, y_train, X_val_prep, y_val, feature_names=feature_names)

    # 7. Learning curves for the best model
    plot_learning_curves(best_model, X_train_prep, y_train, X_val_prep, y_val)

    # 8. Save the model, preprocessor, and thresholds
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(best_model,   os.path.join(model_dir, 'fraud_model.pkl'))
    joblib.dump(preprocessor, os.path.join(model_dir, 'preprocessor.pkl'))
    logger.info("✅ Model and preprocessor saved.")

    thresholds = find_optimal_thresholds(y_val, best_proba)
    with open(os.path.join(model_dir, 'thresholds.json'), 'w') as f:
        json.dump(thresholds, f, indent=2)
    logger.info(f"Thresholds saved: {thresholds}")


if __name__ == '__main__':
    main()