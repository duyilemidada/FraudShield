# routers/predict.py (complete updated version)

from fastapi import APIRouter, Depends, Request, BackgroundTasks
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
router = APIRouter()

@router.post('/predict', response_model=TransactionInDB)
@limiter.limit('1000/hour')
async def predict_and_save(
    request: Request,
    background_tasks: BackgroundTasks,
    transaction: TransactionCreate,
    current_user: User = Depends(get_api_key)
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
    # ── 0. Fetch real‑time velocity features from Redis ──
    velocity = {}
    try:
        # Use customer_email as both the actor and the beneficiary for now
        velocity = get_velocity_features(
            customer_email=transaction.customer_email,
            device_fp=transaction.device_fingerprint or "",
            beneficiary_email=beneficiary
        )
    except Exception as e:
        # If Redis is down, log and continue with zeros – don’t break the request
        client_logger.warning(f"Redis unavailable, using zero velocity features: {e}")
        velocity = {
            "txn_count_5min": 0, "txn_count_1hr": 0, "txn_count_24hr": 0,
            "amount_sum_24hr": 0, "unique_devices_24hr": 0, "inbound_senders_1hr": 0
        }

    anomaly_score = 0.0
    ml_model = request.app.state.ml_model
    anomaly_model = request.app.state.anomaly_model
    X_transformed = None

    if ml_model is not None:
        preproc = ml_model["preprocessor"]
        input_df = pd.DataFrame([{
            'amount':               transaction.amount,
            'log_amount':           np.log1p(transaction.amount),
            'currency':             transaction.currency,
            'payment_method':       transaction.payment_method,
            'transaction_type':     transaction.transaction_type,
            # ── NEW velocity features ─────────────────────
            'txn_count_5min':       velocity['txn_count_5min'],
            'txn_count_1hr':        velocity['txn_count_1hr'],
            'txn_count_24hr':       velocity['txn_count_24hr'],
            'amount_sum_24hr':      velocity['amount_sum_24hr'],
            'unique_devices_24hr':  velocity['unique_devices_24hr'],
            'inbound_senders_1hr':  velocity['inbound_senders_1hr'],
        }])
        X_transformed = preproc.transform(input_df)
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
            # Get SHAP values for the fraud class (class index 1)
            shap_vals = shap_explainer.shap_values(X_transformed)
            if isinstance(shap_vals, list ):
                sv = shap_vals[1][0]  # for binary classification, list of arrays
            else :
                sv = shap_vals[0]

            # Pair feature names with SHAP values
            impacts = list(zip(feature_names, sv))

            # Top 3 by absolute impact
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
                direction  = 'increased' if shap_val > 0 else "decreased"

                reasons.append({
                    "feature":   display,
                    "direction": direction,
                    "impact":    round(abs(float(shap_val)), 4),
                    "text":      f"{display} {direction} fraud risk"
                })


        except Exception as e:
             client_logger.error(f"SHAP explanation failed: {e}")
             
    data['reasons'] = reasons
    saved = await create_transaction(data)

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
            amount=transaction.amount
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

