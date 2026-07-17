# ml/train_fraud_model.py
# End-to-End ML pipeline for FraudShield, following Géron's book
# Loads transactions from MongoDB, engineers features, trains multiple models,
# compares them, diagnoses overfitting via learning curves, and saves the best
# model + preprocessor + decision thresholds for use in the live API.

import pandas as pd
import numpy as np
import asyncio
import logging
import os
import json

from sklearn.feature_selection import SelectFromModel
from ml.feature_stats import save_feature_stats
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,  
)
from defenses.drift_detector import record_and_check_auc
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, VotingClassifier, StackingClassifier,
    )	
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import joblib
import database.mongo as mongo_module   # the same Motor client used by your API

from defenses.training_data_scanner import scan_training_data
from redteam.detect_label_flip import label_consistency_check

TRACKED_CATEGORIES = [
    "purchase", "withdrawal", "transfer",
]

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
    Only rows that actually have a fraud label are kept.
    """
    # ── NEW: Keep only rows where is_fraud is present (not NaN) ──
    df = df.dropna(subset=['is_fraud']).copy()

    drop_cols = [
        '_id', 'transaction_id', 'is_fraud', 'created_at',
        'customer_email', 'customer_phone', 'customer_ip',
        'device_fingerprint', 'merchant_id', 'fraud_score', 'decision', 'recipient_email'
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
    categorical_cols = X.select_dtypes(include=['object', 'str']).columns.tolist()
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

# ═══════════════════════════════════════════════════════════
# Feature selection using SelectFromModel
# ═══════════════════════════════════════════════════════════

def select_features(X_train_prep, y_train, feature_names, threshold="median"):
    """
    Use a Random Forest to score feature importances, then keep only features
    above the threshold importance. This reduces noise and overfitting.

    threshold="median" means: keep features with importance above the median.
    This typically cuts features roughly in half.

    WHY: The book's hub://feature_selection step does something similar.
    In sklearn, SelectFromModel wraps any model with feature_importances_
    and returns a boolean mask of which features to keep.

    Returns:
        X_selected:      reduced feature array
        selector:        fitted SelectFromModel (call .transform() at serving time)
        selected_names:  list of kept feature names (for SHAP)
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel

    logger.info(f"Feature selection: starting with {X_train_prep.shape[1]} features...")

    # Use a fast, shallow RF just for feature importance scoring
    selector_rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    selector_rf.fit(X_train_prep, np.array(y_train))

    selector = SelectFromModel(selector_rf, threshold=threshold, prefit=True)
    X_selected = selector.transform(X_train_prep)

    # Get names of selected features
    if feature_names:
        mask = selector.get_support()
        selected_names = [name for name, keep in zip(feature_names, mask) if keep]
        dropped = [name for name, keep in zip(feature_names, mask) if not keep]
        logger.info(f"  Kept {len(selected_names)} features, dropped {len(dropped)}")
        logger.info(f"  Dropped: {dropped[:10]}{'...' if len(dropped)>10 else ''}")
    else:
        selected_names = None
        logger.info(f"  Kept {X_selected.shape[1]} features (names unavailable)")

    return X_selected, selector, selected_names

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

def log_feature_importance(model, feature_names):
    """Log which features matter most. Works for RF, ExtraTrees, GBM."""
    if not hasattr(model, 'feature_importances_') or not feature_names:
        return
    importances = model.feature_importances_
    feat_imp = sorted(zip(feature_names, importances),
                      key=lambda x: x[1], reverse=True)
    logger.info('=== FEATURE IMPORTANCE ===')
    for name, imp in feat_imp:
        bar = chr(9608) * int(imp * 40)   # block char for visual bar
        logger.info(f'  {name:35s}: {imp:.4f}  {bar}')

def visualise_fraud_cluster(X_prep, y, method='pca'):
    """
    Reduce prepared features to 2D and log cluster separation.
    If fraud and legit points form distinct clusters, your features
    are doing their job. If completely mixed, add more features.
    """
    y_np = np.array(y)

    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)

    X_2d = reducer.fit_transform(X_prep)

    # Separate fraud and legit
    fraud_2d = X_2d[y_np == 1]
    legit_2d = X_2d[y_np == 0]

    # Compute how separated the clusters are
    fraud_centroid = fraud_2d.mean(axis=0)
    legit_centroid = legit_2d.mean(axis=0)
    centroid_distance = np.linalg.norm(fraud_centroid - legit_centroid)

    logger.info(f'=== CLUSTER VISUALISATION ({method.upper()}) ===')
    logger.info(f'  Fraud centroid:  [{fraud_centroid[0]:.3f}, {fraud_centroid[1]:.3f}]')
    logger.info(f'  Legit centroid:  [{legit_centroid[0]:.3f}, {legit_centroid[1]:.3f}]')
    logger.info(f'  Centroid distance: {centroid_distance:.3f}')

    if centroid_distance > 2.0:
        logger.info('  GOOD: Fraud and legit clusters are well separated.')
    elif centroid_distance > 0.5:
        logger.info('  OK: Some separation. More features may help.')
    else:
        logger.warning('  POOR: Clusters overlap heavily. Your features may not be capturing fraud.')

    return X_2d, y_np

def train_unsupervised_anomaly_detectors(X_train, y_train_np, X_val, y_val):
    """
    Train unsupervised anomaly detectors — no labels required.
    These are your day-1 fraud detectors for new merchants,
    and your zero-day fraud catchers for existing merchants.
    """
    results = []

    # ── IsolationForest ──
    logger.info('Training IsolationForest (unsupervised)...')
    fraud_rate = y_train_np.mean() if y_train_np is not None else 0.10
    iso = IsolationForest(n_estimators=100, contamination=float(fraud_rate),
                          random_state=42, n_jobs=-1)
    iso.fit(X_train)

    # Compute bounds from TRAINING data
    train_raw = iso.score_samples(X_train)
    min_s, max_s = train_raw.min(), train_raw.max()

    # Now score validation with those training bounds
    val_raw = iso.score_samples(X_val)
    iso_proba = 1 - (val_raw - min_s) / (max_s - min_s + 1e-9)
    iso_proba = np.clip(iso_proba, 0, 1)

    if y_val is not None:
        auc = roc_auc_score(y_val, iso_proba)
        logger.info(f'  IsolationForest AUC: {auc:.3f}')
        results.append({'name': 'IsolationForest', 'auc': auc, 'model': iso,
                         'proba': iso_proba, 'min_s': min_s, 'max_s': max_s})

    # ── GMM ──
    logger.info('Training GaussianMixture (anomaly detection)...')
    gm = GaussianMixture(n_components=5, n_init=10, random_state=42)
    gm.fit(X_train)

    # Compute density bounds from TRAINING data
    train_log_dens = gm.score_samples(X_train)
    min_d, max_d = train_log_dens.min(), train_log_dens.max()

    val_log_dens = gm.score_samples(X_val)
    gmm_proba = 1 - (val_log_dens - min_d) / (max_d - min_d + 1e-9)
    gmm_proba = np.clip(gmm_proba, 0, 1)

    if y_val is not None:
        auc_gmm = roc_auc_score(y_val, gmm_proba)
        logger.info(f'  GaussianMixture AUC: {auc_gmm:.3f}')
        results.append({'name': 'GMM_Anomaly', 'auc': auc_gmm, 'model': gm,
                         'proba': gmm_proba, 'min_s': min_d, 'max_s': max_d})

    return results
# ═════════════════════════════════════════════════════════
# 8. TRAIN MULTIPLE MODELS AND PICK THE BEST
# ═════════════════════════════════════════════════════════
def train_and_compare(X_train, y_train, X_val, y_val, feature_names=None):
    """
    Trains multiple models + ensembles, returns best by ROC-AUC:
      A) LogReg_Ridge
      B) LogReg_Lasso
      C) DecisionTree
      D) SVM_RBF
      E) RandomForest (with OOB + feature importance)
      F) ExtraTrees
      G) GradientBoosting
      H) XGBoost                    ← NEW
      I) VotingClassifier (soft)
      J) StackingClassifier         ← NEW
    """
    results = []
    y_train_np = np.array(y_train)
    sample_weights = compute_sample_weight('balanced', y_train_np)

    # ── A: LogReg Ridge ──────────────────────────────────────────────
    logger.info('Training LogReg_Ridge...')
    lr = LogisticRegression(C=1.0, solver='lbfgs',
                             max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train_np)
    lr_proba = lr.predict_proba(X_val)[:, 1]
    results.append(evaluate_model('LogReg_Ridge', y_val, lr.predict(X_val), lr_proba))

    # ── B: LogReg Lasso ──────────────────────────────────────────────
    logger.info('Training LogReg_Lasso...')
    lr_l1 = LogisticRegression(C=0.1,penalty='l1' , solver='liblinear',
                                class_weight='balanced', random_state=42)
    lr_l1.fit(X_train, y_train_np)
    logger.info(f'  Lasso zeroed {np.sum(lr_l1.coef_[0]==0)} features.')
    l1_proba = lr_l1.predict_proba(X_val)[:, 1]
    results.append(evaluate_model('LogReg_Lasso', y_val, lr_l1.predict(X_val), l1_proba))

    # ── C: Decision Tree ─────────────────────────────────────────────
    logger.info('Training DecisionTree...')
    dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5,
                                 min_samples_split=10, class_weight='balanced',
                                 random_state=42)
    dt.fit(X_train, y_train_np)
    dt_proba = dt.predict_proba(X_val)[:, 1]
    results.append(evaluate_model('DecisionTree', y_val, dt.predict(X_val), dt_proba))
    if feature_names:
        logger.info('Decision Tree rules:\n' + export_text(dt, feature_names=list(feature_names)))

    # ── D: SVM RBF ───────────────────────────────────────────────────
    logger.info('Training SVM_RBF...')
    svm_base = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    svm_grid = GridSearchCV(svm_base,
        {'C':[0.1,1,10,100], 'gamma':['scale','auto',0.01,0.1]},
        cv=3, scoring='roc_auc', n_jobs=-1)
    svm_grid.fit(X_train, y_train_np)
    best_svm = svm_grid.best_estimator_
    logger.info(f'  Best SVM params: {svm_grid.best_params_}')
    svm_proba = best_svm.predict_proba(X_val)[:, 1]
    results.append(evaluate_model('SVM_RBF', y_val, best_svm.predict(X_val), svm_proba))

    # ── E: Random Forest ─────────────────────────────────────────────
    logger.info('Training RandomForest...')
    rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                 class_weight='balanced', oob_score=True,
                                 n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train_np)
    logger.info(f'  OOB score: {rf.oob_score_:.3f}')
    log_feature_importance(rf, feature_names)
    rf_proba = rf.predict_proba(X_val)[:, 1]
    results.append(evaluate_model('RandomForest', y_val, rf.predict(X_val), rf_proba))

    # ── F: Extra-Trees ───────────────────────────────────────────────
    logger.info('Training ExtraTrees...')
    et = ExtraTreesClassifier(n_estimators=100, max_depth=10,
                               class_weight='balanced', n_jobs=-1, random_state=42)
    et.fit(X_train, y_train_np)
    et_proba = et.predict_proba(X_val)[:, 1]
    results.append(evaluate_model('ExtraTrees', y_val, et.predict(X_val), et_proba))

    # ── G: Gradient Boosting ─────────────────────────────────────────
    logger.info('Training GradientBoosting...')
    gbm = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                      learning_rate=0.1, subsample=0.8,
                                      random_state=42)
    gbm.fit(X_train, y_train_np, sample_weight=sample_weights)
    log_feature_importance(gbm, feature_names)
    gbm_proba = gbm.predict_proba(X_val)[:, 1]
    results.append(evaluate_model('GradientBoosting', y_val, gbm.predict(X_val), gbm_proba))

 # ── H: XGBoost ─────────────────────────────────────────────
    logger.info('Training XGBoost...')
    scale_pos_weight = (y_train_np == 0).sum() / max((y_train_np == 1).sum(), 1)

    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        early_stopping_rounds=10,
        random_state=42
    )
    xgb_clf.fit(
        X_train, y_train_np,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    logger.info(f'  Best iteration: {xgb_clf.best_iteration}')

    # Retrim the model to the optimal number of trees
    best_iter = xgb_clf.best_iteration
    if best_iter < 200:
        xgb_clf_trimmed = xgb.XGBClassifier(
            n_estimators=best_iter,
            max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
            eval_metric='auc', random_state=42
        )
        xgb_clf_trimmed.fit(X_train, y_train_np)
        xgb_clf = xgb_clf_trimmed
        logger.info(f"  Trimmed XGBoost to {best_iter} trees")
    else:
        logger.info(f"  No trimming needed (best_iter == 200)")

    xgb_proba = xgb_clf.predict_proba(X_val)[:, 1]
    results.append(evaluate_model('XGBoost', y_val, xgb_clf.predict(X_val), xgb_proba))
    
    # ── I: Soft VotingClassifier (existing) ──────────────────────────
    logger.info('Training VotingClassifier (soft)...')
    voting = VotingClassifier(
        estimators=[('lr',lr),('dt',dt),('svm',best_svm),('rf',rf),('gbm',gbm)],
        voting='soft', n_jobs=-1
    )
    voting.fit(X_train, y_train_np)
    voting_proba = voting.predict_proba(X_val)[:, 1]
    results.append(evaluate_model('VotingClassifier', y_val, voting.predict(X_val), voting_proba))

    # ── J: StackingClassifier (NEW) ──────────────────────────────────
    logger.info('Training StackingClassifier...')
    # Base estimators: we reuse the already trained models for speed
    stacking = StackingClassifier(
        estimators=[
            ('lr', lr), ('dt', dt), ('svm', best_svm), ('rf', rf), ('gbm', gbm)
        ],
        final_estimator=LogisticRegression(C=1.0, class_weight='balanced'),
        cv=5,           # 5-fold CV to generate training data for the blender
        n_jobs=-1
    )
    # Note: StackingClassifier.fit will clone the base estimators and fit them again.
    # That's fine – we want it to learn on the full training data with cross-validation.
    stacking.fit(X_train, y_train_np)
    stacking_proba = stacking.predict_proba(X_val)[:, 1]
    results.append(evaluate_model('StackingClassifier', y_val, stacking.predict(X_val), stacking_proba))

    # ── Pick best ─────────────────────────────────────────────────────
    best = max(results, key=lambda r: r['auc'])
    logger.info(f'✅ BEST MODEL: {best["name"]} (AUC={best["auc"]:.3f})')

    models = {
        'LogReg_Ridge':       lr,
        'LogReg_Lasso':       lr_l1,
        'DecisionTree':       dt,
        'SVM_RBF':            best_svm,
        'RandomForest':       rf,
        'ExtraTrees':         et,
        'GradientBoosting':   gbm,
        'XGBoost':            xgb_clf,
        'VotingClassifier':   voting,
        'StackingClassifier': stacking,
    }
    probas = {
        'LogReg_Ridge':       lr_proba,
        'LogReg_Lasso':       l1_proba,
        'DecisionTree':       dt_proba,
        'SVM_RBF':            svm_proba,
        'RandomForest':       rf_proba,
        'ExtraTrees':         et_proba,
        'GradientBoosting':   gbm_proba,
        'XGBoost':            xgb_proba,
        'VotingClassifier':   voting_proba,
        'StackingClassifier': stacking_proba,
    }
    return models[best['name']], probas[best['name']]
    
def semi_supervised_learning(X_prep, y_full=None, label_fraction=0.1, k=50):
    """
    Simulate the semi-supervised workflow.
    - X_prep : preprocessed training features (numpy array)
    - y_full : optional full labels (for testing with synthetic data)
    - label_fraction : fraction of data to treat as 'labelled' (simulates sparse chargebacks)
    - k : number of clusters

    Returns:
        propagated_labels : labels for all training samples (after propagation)
        train_mask : boolean mask of samples that ended up in the propagation set
                     (only the closest 20% in each cluster, as recommended by the book)
    """


    logger.info(f"=== SEMI-SUPERVISED LEARNING (k={k}) ===")
    n = len(X_prep)

    # Step 1: cluster all data
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_prep)
    X_dist = kmeans.transform(X_prep)          # distances to each centroid

    # Step 2: find the most representative sample per cluster (closest to centroid)
    representative_idx = np.argmin(X_dist, axis=0)  # shape (k,)

    # Step 3: get labels for those representatives
    if y_full is not None:
        # Synthetic mode: we have all labels, but we pretend only a few are labelled
        # Take the representative indices, and optionally also a random subset to
        # simulate a real scenario where you might have a few extra labels.
        reps_labels = y_full[representative_idx].copy()
        # Simulate "only label_fraction of the data is known"
        known_mask = np.zeros(n, dtype=bool)
        known_mask[representative_idx] = True   # these are definitely known
        # Add a few random labelled examples to mimic real randomness

        num_desired = int(n * label_fraction)
        if num_desired > k:


            unlabeled = np.where(~known_mask)[0]
            extra = np.random.choice(unlabeled, size=int(n * label_fraction) - k, replace=False)
            known_mask[extra] = True
            logger.info(f"  Using {known_mask.sum()} labelled samples (representatives + random)")
        else :
            # label_fraction is too small – just use the representatives
            logger.warning(f"  Requested {num_desired} labelled samples, but need at least {k} "
                           f"for {k} clusters. Using only the {k} representatives.")
    else:
        # Real production mode: you have a mask of known labels (e.g., chargebacks)
        # In that case, known_mask would be passed in. We'll just use representatives.
        # For simplicity, if y_full is None, we raise an error (you can extend later)
        raise ValueError("For real data, pass in y_full with known indices set and others = -1")
    
    # Step 4 (new): Determine cluster label by majority vote of known samples
    cluster_labels = {}
    for i in range(k):
        cluster_mask = kmeans.labels_ == i
        # Get labels of known samples in this cluster
        known_in_cluster = y_full[cluster_mask & known_mask]
        if len(known_in_cluster) > 0:
            # Majority vote (if tie, choose 0 – conservative)
            majority = int(known_in_cluster.sum() >= len(known_in_cluster) / 2)
        else :
            # No known sample in cluster: fall back to representative
            majority = reps_labels[i]
        cluster_labels[i] = majority
    # Step 4: Propagate labels inside each cluster
    propagated = np.empty(n, dtype=int)
    
    for i in range(k):
        cluster_mask = kmeans.labels_ == i
        # The representative's label becomes the label for the whole cluster
        propagated[cluster_mask] = cluster_labels[i]

    # Step 5 (optional but recommended): keep only the closest 20% in each cluster
    percentile = 20
    # Distance of each point to its own cluster's centroid
    own_dist = X_dist[np.arange(n), kmeans.labels_]
    keep_mask = np.zeros(n, dtype=bool)
    for i in range(k):
        in_cluster = kmeans.labels_ == i
        if in_cluster.sum() == 0:
            continue
        cutoff = np.percentile(own_dist[in_cluster], percentile)
        keep_mask[in_cluster] = own_dist[in_cluster] <= cutoff

    propagated_partial = propagated[keep_mask]
    # Note: we also need to keep only those samples that were actually in the "known" mask?
    # Typically you propagate from known labels, so the keep_mask already filters.
    # We'll just use keep_mask as our training set mask.

    logger.info(f"  Propagated labels to {keep_mask.sum()} samples "
                f"({keep_mask.sum()/n:.1%} of training data)")

    # Optional: show label distribution
    unique, counts = np.unique(propagated_partial, return_counts=True)
    logger.info(f"  Propagated label distribution: {dict(zip(unique, counts))}")

    return propagated_partial, keep_mask

# ═════════════════════════════════════════════════════════
# Feature: Historical velocity simulation for training
# ═════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════
# Feature: Historical velocity simulation for training
# ═════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

def add_training_velocity_features(df):
    """
    Compute velocity features from historical data in time order.
    Uses an O(n²) loop per customer — fast enough for <10,000 rows.

    SKEW AUDIT: Every feature here must be computed identically in redis_client.py.
    The comments mark how each feature maps to its Redis equivalent.
    """
    if len(df) > 10_000:
        logger.warning(
            f"add_training_velocity_features: {len(df)} rows — O(n²) loop will be slow. "
            f"Consider a merge-asof based approach at this scale."
        )

    df = df.sort_values('created_at').copy()
    df['created_at'] = pd.to_datetime(df['created_at'], utc=True, errors='coerce')

    # Initialise all velocity columns to 0
    df['txn_count_5min']      = 0
    df['txn_count_1hr']       = 0
    df['txn_count_24hr']      = 0
    df['amount_sum_24hr']     = 0.0
    df['unique_devices_24hr'] = 0
    df['inbound_senders_1hr'] = 0

    # Initialise category columns
    # Redis equivalent: cust:{email}:cat:{category} sorted set, 14d window
    for cat in TRACKED_CATEGORIES:
        df[f'category_{cat}_count_14d'] = 0

    # ── Part 1: Customer-level features ──────────────────────────────────────
    for email, group in df.groupby('customer_email'):
        if len(group) <= 1:
            continue
        group = group.sort_values('created_at')
        # Convert to unix seconds for arithmetic
        times = group['created_at'].astype('int64') // 1_000_000_000

        for i in range(1, len(group)):
            current_ts  = times.iloc[i]
            prev_times  = times.iloc[:i]          # strictly before current txn

            # Count windows — Redis: zcount key (now-window) now
            df.at[group.index[i], 'txn_count_5min']  = ((current_ts - prev_times) <= 300).sum()
            df.at[group.index[i], 'txn_count_1hr']   = ((current_ts - prev_times) <= 3_600).sum()
            df.at[group.index[i], 'txn_count_24hr']  = ((current_ts - prev_times) <= 86_400).sum()

            # Amount sum — Redis: zrangebyscore amounts (now-86400) now → sum values
            in_24h = (current_ts - prev_times) <= 86_400
            df.at[group.index[i], 'amount_sum_24hr'] = group['amount'].iloc[:i][in_24h].sum()

            # Unique devices — Redis: zrangebyscore devices (now-86400) now → len(set())
            # NOTE: we now count unique device values in the 24h window — matching
            # the FIXED redis_client.py which uses a sorted set + zrangebyscore.
            if 'device_fingerprint' in group.columns:
                devs_in_24h = group['device_fingerprint'].iloc[:i][in_24h]
                df.at[group.index[i], 'unique_devices_24hr'] = devs_in_24h.nunique()

            # Category counts — Redis: zcount cust:email:cat:X (now-14d) now
            in_14d = (current_ts - prev_times) <= 1_209_600
            if 'transaction_type' in group.columns:
                prev_types = group['transaction_type'].iloc[:i]
                for cat in TRACKED_CATEGORIES:
                    cat_col = f'category_{cat}_count_14d'
                    count = ((prev_types == cat) & in_14d).sum()
                    df.at[group.index[i], cat_col] = int(count)

    # ── Part 2: Beneficiary-level feature (mule detection) ───────────────────
    if 'recipient_email' not in df.columns or not df['recipient_email'].notna().any():
        logger.info("No recipient_email — inbound_senders_1hr stays 0.")
        return df

    times_sec = df['created_at'].astype('int64') // 1_000_000_000
    has_recipient = df['recipient_email'].notna() & (df['recipient_email'] != '')

    for idx, row in df[has_recipient].iterrows():
        recipient    = row['recipient_email']
        current_ts   = times_sec[idx]
        window_start = current_ts - 3_600

        mask = (
            (df['recipient_email'] == recipient) &
            (times_sec < current_ts) &          # strictly before
            (times_sec >= window_start)
        )
        df.at[idx, 'inbound_senders_1hr'] = df.loc[mask, 'customer_email'].nunique()

    logger.info("Velocity features (including categories) computed.")
    return df

# ═════════════════════════════════════════════════════════
# 9. MAIN PIPELINE
# ═════════════════════════════════════════════════════════
def main():
    # 1. Load
    df = load_data()
    if df.empty:
        logger.error("No data found. Run: python -m ml.seed_synthetic_data")
        return

    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)

    # ── 1b. Quarantine defense ────────────────────────────────────────────────
    if '_quarantined' in df.columns:
        before = len(df)
        df = df[df['_quarantined'] != True]
        logger.info(f"Removed {before - len(df)} quarantined rows")

    # ── 2. Temporal split (no future leakage) ─────────────────────────────────
    df = df.sort_values('created_at')
    cutoff = int(len(df) * 0.8)
    train_df = df.iloc[:cutoff].copy()
    val_df   = df.iloc[cutoff:].copy()
    logger.info(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")

    # ── 3. Velocity features (in time order) ──────────────────────────────────
    combined = pd.concat([train_df, val_df], axis=0)
    combined = add_training_velocity_features(combined)
    train_df = combined.iloc[:len(train_df)]
    val_df   = combined.iloc[len(train_df):]

    # ── 4. Feature engineering ────────────────────────────────────────────────
    train_df = engineer_features(train_df)
    val_df   = engineer_features(val_df)

    # ── 5. Split X/y ──────────────────────────────────────────────────────────
    X_train, y_train = split_features_labels(train_df)
    X_val,   y_val   = split_features_labels(val_df)
    logger.info(f"Fraud rate in training: {y_train.mean():.1%}")
    logger.info(f"Features: {list(X_train.columns)}")

    # ── 6. Save feature stats for serving-time imputation ────────────────────
    # Implements the book's impute_policy={"*": "$mean"}
    from ml.feature_stats import save_feature_stats
    save_feature_stats(X_train, model_dir)

    # ── 7. Save reference distribution for drift detection ───────────────────
    from ml.data_drift import save_reference_distribution
    save_reference_distribution(X_train, model_dir)

    # ── 8. Preprocessing pipeline ─────────────────────────────────────────────
    preprocessor = build_preprocessing_pipeline(X_train)
    X_train_prep = preprocessor.fit_transform(X_train)
    X_val_prep   = preprocessor.transform(X_val)

    # Build feature names for SHAP and feature selection
    try:
        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_train.select_dtypes(include=['object', 'string']).columns.tolist()
        ohe = preprocessor.named_transformers_['cat']['onehot']
        cat_names = ohe.get_feature_names_out(cat_cols).tolist()
        feature_names = num_cols + cat_names
    except Exception as e:
        logger.warning(f"Could not extract feature names: {e}")
        feature_names = None

    # ── 9. Cluster visualisation (for EDA insight) ───────────────────────────
    y_train_np = np.array(y_train)
    visualise_fraud_cluster(X_train_prep, y_train, method='pca')

    # ── 10. Training data defenses ────────────────────────────────────────────
    outlier_idx    = scan_training_data(X_train_prep, contamination=0.05)
    label_flip_idx = label_consistency_check(X_train_prep, y_train, k=10, threshold=0.80)
    suspect_indices = np.unique(np.concatenate([outlier_idx, label_flip_idx]))
    logger.info(f"Removing {len(suspect_indices)} suspect samples")
    X_clean = np.delete(X_train_prep, suspect_indices, axis=0)
    y_clean = np.delete(y_train_np, suspect_indices)

    # ── 11. Feature selection ─────────────────────────────────────────────────
    X_clean_selected, feature_selector, selected_names = select_features(
        X_clean, y_clean, feature_names, threshold="median"
    )
    X_val_selected = feature_selector.transform(X_val_prep)
    joblib.dump(feature_selector, os.path.join(model_dir, 'feature_selector.pkl'))

    # ── 12. Semi-supervised experiment ───────────────────────────────────────
    propagated_labels, train_mask = semi_supervised_learning(
        X_clean_selected,
        y_full=y_clean,
        label_fraction=0.1,
        k=min(50, len(X_clean_selected))
    )
    lr_semi = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr_semi.fit(X_clean_selected[train_mask], propagated_labels)
    semi_auc = roc_auc_score(y_val, lr_semi.predict_proba(X_val_selected)[:, 1])
    logger.info(f"Semi-supervised LogReg AUC (10% labels): {semi_auc:.3f}")

    # ── 13. Train supervised models ───────────────────────────────────────────
    best_model, best_proba = train_and_compare(
        X_clean_selected, y_clean, X_val_selected, y_val,
        feature_names=selected_names
    )
    best_auc = roc_auc_score(y_val, best_proba)

    # ── 14. AUC drift check ───────────────────────────────────────────────────
    alarm = record_and_check_auc(best_auc)
    if alarm:
        logger.warning("🚨 AUC drift detected — possible incremental poisoning!")

    # ── 15. Unsupervised anomaly detectors ────────────────────────────────────
    unsupervised_results = train_unsupervised_anomaly_detectors(
        X_clean_selected, y_clean, X_val_selected, np.array(y_val)
    )
    if unsupervised_results:
        best_unsup = max(unsupervised_results, key=lambda r: r['auc'])
        joblib.dump(best_unsup['model'], os.path.join(model_dir, 'anomaly_model.pkl'))
        with open(os.path.join(model_dir, 'anomaly_bounds.json'), 'w') as f:
            json.dump({'min_s': best_unsup['min_s'], 'max_s': best_unsup['max_s'],
                       'model_type': best_unsup['name']}, f)

    # ── 16. Save model and supporting files ───────────────────────────────────
    joblib.dump(best_model,   os.path.join(model_dir, 'fraud_model.pkl'))
    joblib.dump(preprocessor, os.path.join(model_dir, 'preprocessor.pkl'))

    thresholds = find_optimal_thresholds(y_val, best_proba)
    with open(os.path.join(model_dir, 'thresholds.json'), 'w') as f:
        json.dump(thresholds, f, indent=2)

    if selected_names:
        with open(os.path.join(model_dir, 'feature_names.json'), 'w') as f:
            json.dump(selected_names, f)

    from datetime import datetime, timezone
    version_str = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    with open(os.path.join(model_dir, 'model_version.txt'), 'w') as f:
        f.write(version_str)

    logger.info(f"✅ Training complete. Model version: {version_str}")
    logger.info(f"   AUC: {best_auc:.3f}")
    logger.info(f"   Block threshold: {thresholds['BLOCK_THRESHOLD']:.4f}")
    logger.info(f"   Review threshold: {thresholds['REVIEW_THRESHOLD']:.4f}")


if __name__ == '__main__':
    main()