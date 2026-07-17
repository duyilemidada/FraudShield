# routers/predict.py (complete updated version)
from ml.feature_stats import impute_with_stats
from services.redis_client import TRACKED_CATEGORIES
from fastapi import APIRouter, Depends, Request, BackgroundTasks, HTTPException
from schemas.transaction import TransactionCreate, TransactionInDB
from services.fraud_service import calculate_fraud_score
from crud.transaction_crud import create_transaction
from crud.security import get_api_key
from schemas.users import User
from logger_config import client_logger
import pandas as pd
import numpy as np
from crud.get_current_user import role_required
from schemas.users import Role
from rate_limiter import limiter
from background_tasks import send_fraud_alert_webhook, log_to_analytics
from services.redis_client import get_velocity_features, record_transaction
from services.audit_logger import write_audit_log
import time
import os
from pymongo.errors import DuplicateKeyError
from middleware.validation import validate_transaction_request
router = APIRouter()

@router.post('/predict', response_model=TransactionInDB)
@limiter.limit('1000/hour')
async def predict_and_save(
    request: Request,
    background_tasks: BackgroundTasks,
    transaction: TransactionCreate,
    current_user: User = Depends(get_api_key),
    _valid : None = Depends(validate_transaction_request)
):
    start_time = time.perf_counter()

    data = transaction.model_dump()
    # ── Capture raw features BEFORE adding merchant_id ──
    raw_features = transaction.model_dump(mode='json')

    data['merchant_id'] = str(current_user.id)

    client_logger.info(
        f'Predict request: merchant={current_user.id}, '
        f'txn={transaction.transaction_id}, amount={transaction.amount}'
    )

    beneficiary = transaction.recipient_email or transaction.customer_email
    # ── 0. Fetch real-time velocity features from Redis ─────────────────────
    feature_stats = getattr(request.app.state, 'feature_stats', {})

    try:
        beneficiary = transaction.recipient_email or transaction.customer_email
        velocity = get_velocity_features(
            customer_email=transaction.customer_email,
            device_fp=transaction.device_fingerprint or "",
            beneficiary_email=beneficiary
        )
    except Exception as e:
        client_logger.warning(f"Redis unavailable — using training-mean imputation: {e}")
        # Build a dict of None values so impute_with_stats fills them with means
        velocity = {
            "txn_count_5min":      None,
            "txn_count_1hr":       None,
            "txn_count_24hr":      None,
            "amount_sum_24hr":     None,
            "unique_devices_24hr": None,
            "inbound_senders_1hr": None,
        }
        for cat in TRACKED_CATEGORIES:
            velocity[f"category_{cat}_count_14d"] = None


    anomaly_score = 0.0
    ml_model = request.app.state.ml_model
    anomaly_model = request.app.state.anomaly_model
    X_transformed = None

    # Apply statistical imputation (training-set means replace None values).
    # For a brand-new customer with no Redis history, this is fine —
    # all their values are 0, not None, because zcount on a missing key returns 0.
    # Imputation only kicks in when Redis itself is down.
    velocity = impute_with_stats(velocity, feature_stats, strategy="mean")

    # ── 1. Build input DataFrame for the preprocessor ────────────────────────
    if ml_model is not None:
        preproc = ml_model["preprocessor"]

        # Base features from the transaction
        row = {
            'amount':           transaction.amount,
            'log_amount':       np.log1p(transaction.amount),
            'currency':         transaction.currency,
            'payment_method':   transaction.payment_method,
            'transaction_type': transaction.transaction_type,
        }

        # Merge in all velocity features (including category counts)
        row.update(velocity)

        input_df = pd.DataFrame([row])

        # ── Step 1: preprocessor (scaling + one-hot) ─────────────────────────
        X_prep = preproc.transform(input_df)

        # ── Step 2: feature selector (keep only training-selected features) ──
        feature_selector = ml_model.get('feature_selector')
        if feature_selector is not None:
            try:
                X_transformed = feature_selector.transform(X_prep)
            except Exception as e:
                client_logger.warning(f"Feature selector failed: {e}. Using full features.")
                X_transformed = X_prep
        else:
            X_transformed = X_prep
    else:
        X_transformed = None

    # ── 2. Unsupervised anomaly score ──
    if anomaly_model is not None and X_transformed is not None:
        try:
            raw = anomaly_model['model'].score_samples(X_transformed)[0]
            b = anomaly_model['bounds']
            anomaly_score = float(1.0 - (raw - b['min_s']) / (b['max_s'] - b['min_s'] + 1e-9))
            anomaly_score = max(0.0, min(1.0, anomaly_score))
        except Exception as e:
            client_logger.warning(f'Anomaly scoring failed: {e}')

    # ── 3. Supervised ML scoring ──
    fraud_proba = None
    decision = None
    score = None

    if ml_model is not None and X_transformed is not None:
        try:
            fraud_proba = ml_model['classifier'].predict_proba(X_transformed)[0][1]
            combined_score = max(fraud_proba, anomaly_score * 0.7)
            score = float(round(combined_score * 100, 2))

            thresholds = ml_model.get('thresholds', {"BLOCK_THRESHOLD": 0.75, "REVIEW_THRESHOLD": 0.35})
            block_thresh = thresholds.get('BLOCK_THRESHOLD', 0.75)
            review_thresh = thresholds.get('REVIEW_THRESHOLD', 0.35)

            if fraud_proba >= block_thresh:
                decision = 'block'
            elif fraud_proba >= review_thresh:
                decision = 'review'
            else:
                decision = 'approve'

            client_logger.info(
                f'ML: score={score:.1f}, decision={decision}, '
                f'merchant={current_user.id}, anomaly={anomaly_score:.3f}'
            )
        except Exception as e:
            client_logger.error(f'ML failed: {e}. Using rules.')
            fraud_proba = None

    # ── 4. Fallback ──
    if fraud_proba is None:
        if anomaly_score > 0.8:
            decision = 'review'
            score = round(anomaly_score * 100, 2)
            client_logger.info(f'Anomaly-only: score={score}, decision=review')
        else:
            score, decision = calculate_fraud_score(data)
            client_logger.info(f'Rule-based: score={score}, decision={decision}')

    data['fraud_score'] = score
    data['decision'] = decision
    # ── SHAP explainability ─────────────────────────────────
    reasons = []
    shap_explainer = ml_model.get('shap_explainer') if ml_model else None
    feature_names = ml_model.get('feature_names') if ml_model else None

    if shap_explainer is not None and feature_names is not None and X_transformed is not None:
        try:
            # shap_values can return either:
            #  - a single array of shape (n_samples, n_features)  [most common for binary]
            #  - a list of arrays [one per class] for classifiers
            shap_raw = shap_explainer.shap_values(X_transformed)

            if isinstance(shap_raw, list):
                # list of arrays → use the fraud class (index 1)
                sv = shap_raw[1]
            else:
                sv = shap_raw

            # sv shape should be (1, n_features) or (n_features,)
            sv = np.squeeze(np.array(sv))

            # After squeeze, if there's still more than one dimension, take the first row
            if sv.ndim > 1:
                sv = sv[0]

            # Now sv is 1‑D with length = n_features
            impacts = list(zip(feature_names, sv))
            top = sorted(impacts, key=lambda x: abs(x[1]), reverse=True)[:3]
            # Human‑readable labels for common features
            readable_map = {
                'amount': 'Transaction amount',
                'log_amount': 'Transaction amount scale',
                'txn_count_5min': 'Transactions in last 5 minutes',
                'txn_count_1hr': 'Transactions in last hour',
                'txn_count_24hr': 'Transactions in last 24 hours',
                'amount_sum_24hr': 'Total amount sent in last 24 hours',
                'unique_devices_24hr': 'Devices used in last 24 hours',
                'inbound_senders_1hr': 'Senders to this account in last hour',
            }

            for feat_name, shap_val in top:
                # Convert one‑hot names like payment_method_ussd → "Payment method USSD"
                
                display = feat_name
                for prefix, label in readable_map.items():
                    if feat_name.startswith(prefix):
                        display = label
                        break
                if display == feat_name:
                    # Clean up one‑hot names if no mapping found
                    if '_' in display:
                        display = display.replace('_', ' ').title()
                direction  = 'increased' if float(shap_val) > 0 else "decreased"

                reasons.append({
                    "feature":   display,
                    "direction": direction,
                    "impact":    round(abs(float(shap_val)), 4),
                    "text":      f"{display} {direction} fraud risk"
                })


        except Exception as e:
             client_logger.error(f"SHAP explanation failed: {e}")
             
    data['reasons'] = reasons
    try:
      saved = await create_transaction(data)
    except DuplicateKeyError:
        raise HTTPException(
            status_code=409,
            detail=f"Transaction '{transaction.transaction_id}' already exists. Use a unique transaction_id."
        )

    # ── Write immutable audit log (synchronous, must succeed) ──
    processing_time = (time.perf_counter() - start_time) * 1000
    try:
        await write_audit_log(
            transaction_id=transaction.transaction_id,
            merchant_id=str(current_user.id),
            fraud_score=score,
            decision=decision,
            model_version=request.app.state.model_version,
            features_used=raw_features,
            reasons=reasons,                     # from SHAP
            processing_ms=processing_time
        )
    except Exception as e :
        client_logger.error(f"Audit log write failed: {e}")
    # ── 5. Record this transaction in Redis for future velocity queries ──
    try:
        record_transaction(
            customer_email=transaction.customer_email,
            device_fp=transaction.device_fingerprint or "",
            beneficiary_email=beneficiary,
            txn_id=transaction.transaction_id,
            amount=transaction.amount,
            transaction_type=transaction.transaction_type  # NEW
        )
    except Exception as e:
        client_logger.warning(f"Failed to record transaction in Redis: {e}")

    # Queue background tasks
    if data.get('decision') == 'block':
        background_tasks.add_task(
            send_fraud_alert_webhook,
            merchant_webhook_url=getattr(current_user, 'webhook_url', None),
            transaction_id=transaction.transaction_id,
            fraud_score=data["fraud_score"],
            decision="block"
        )
    background_tasks.add_task(log_to_analytics, data)

    return saved



def _describe_policy(thresholds):
    block = thresholds.get('BLOCK_THRESHOLD', 0.75)
    review = thresholds.get('REVIEW_THRESHOLD', 0.35)
    if block >= 0.80:
        return 'conservative'  # only very confident fraud is blocked
    elif block >= 0.60:
        return 'balanced'
    else:
        return 'aggressive'




@router.get('/admin/drift')
async def check_drift(
    request: Request,
    current_user: User = Depends(role_required(Role.ADMIN))
):
    """
    Run a drift check on the last 500 transactions.
    Compare their feature distributions to the training-set reference.
    Returns PSI score per feature with GREEN/YELLOW/RED status.

    Run this weekly or after any data pipeline change.
    PSI > 0.25 on velocity features means your Redis patterns have shifted
    significantly from what the model was trained on.
    """
    import database.mongo as mongo_module
    from ml.data_drift import compute_drift_report
    import pandas as pd

    # Fetch recent transactions (last 500)
    docs = []
    async for doc in mongo_module.transaction_collection.find(
        {}, {"_id": 0}
    ).sort("created_at", -1).limit(500):
        docs.append(doc)

    if not docs:
        return {"error": "No transactions found for drift analysis."}

    recent_df = pd.DataFrame(docs)
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml', 'models')

    report = compute_drift_report(recent_df, model_dir)
    return report

@router.get('/model-info')
async def model_info(
    request: Request,
    current_user: User = Depends(role_required(Role.ADMIN))
):
    """
    Internal endpoint — returns model diagnostics for admins only.
    Never expose exact thresholds to merchants.
    """
    ml_model = request.app.state.ml_model
    if ml_model is None:
        return {'mode': 'rule-based', 'description': 'No Ml model loaded.'}
    model_type = type(ml_model['classifier']).__name__
    thresholds = ml_model.get('thresholds', {})

    return {
        'mode': 'ml',
        'model_type': model_type,
        # Return threshold TIERS not exact values:
        # Merchant sees: 'aggressive' / 'balanced' / 'conservative'
        # Not: 0.75 and 0.35
        'risk_policy': _describe_policy(thresholds)
    }

