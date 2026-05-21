
from fastapi import APIRouter, Depends, Request, BackgroundTasks
from schemas.transaction import TransactionCreate, TransactionInDB
from services.fraud_service import calculate_fraud_score
from crud.transaction_crud import create_transaction
from crud.security import get_api_key
from schemas.users import User
from logger_config import client_logger
import pandas as pd
from crud.get_current_user import role_required
from schemas.users import Role
from rate_limiter import limiter
import numpy as np
from background_tasks import send_fraud_alert_webhook, log_to_analytics
# Thresholds based on precision/recall tradeoff 
BLOCK_THRESHOLD  = 0.75   # very confident → auto-block
REVIEW_THRESHOLD = 0.35   # moderately suspicious → human review

router = APIRouter()

@router.post('/predict', response_model=TransactionInDB)
@limiter.limit('1000/hour')
async def predict_and_save(
    request: Request,
    background_tasks: BackgroundTasks,
    transaction: TransactionCreate,
    current_user: User = Depends(get_api_key)
):
    data = transaction.model_dump()
    data['merchant_id'] = str(current_user.id)

    client_logger.info(
        f'Predict request: merchant={current_user.id}, '
        f'txn={transaction.transaction_id}, amount={transaction.amount}'
    )

    anomaly_score = 0.0

    # ── 1. Preprocess the transaction (needed by any model) ──
    ml_model = request.app.state.ml_model
    anomaly_model = request.app.state.anomaly_model
    X_transformed = None

    # Preprocessor is stored inside ml_model (same for both)
    if ml_model is not None :
        # Use ml_model's preprocessor (both models were trained on the same pipeline)
        preproc = ml_model["preprocessor"] 
        input_df = pd.DataFrame([{
            'amount':           transaction.amount,
            'log_amount':       np.log1p(transaction.amount), 
            'currency':         transaction.currency,
            'payment_method':   transaction.payment_method,
            'transaction_type': transaction.transaction_type,
        }])
        X_transformed = preproc.transform(input_df)
    else :
        X_transformed = None

    # ── 2. Unsupervised anomaly score ──
    
    if anomaly_model is not None and X_transformed is not None:
        try:
            raw = anomaly_model['model'].score_samples(X_transformed)[0]
            b = anomaly_model['bounds']
            # Normalise to 0 (normal) → 1 (anomalous)
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

            # Blend with anomaly score (70% weight for anomaly)
            combined_score = max(fraud_proba, anomaly_score * 0.7)
            score = float(round(combined_score * 100, 2))

            # Use business thresholds
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
            fraud_proba = None   # force fallback

    # ── 4. Fallback: rule-based or anomaly-only ──
    if fraud_proba is None:
        # No supervised model – use anomaly or rules
        if anomaly_score > 0.8:   # strong anomaly, no supervised decision
            decision = 'review'
            score = round(anomaly_score * 100, 2)
            client_logger.info(f'Anomaly-only: score={score}, decision=review')
        else:
            # Pure rule-based fallback
            score, decision = calculate_fraud_score(data)
            client_logger.info(f'Rule-based: score={score}, decision={decision}')

    data['fraud_score'] = score
    data['decision'] = decision

    saved =  await create_transaction(data)

    # Queue background tasks (non-blocking — run after response)
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

