
from fastapi import APIRouter, Depends, Request
from schemas.transaction import TransactionCreate, TransactionInDB
from services.fraud_service import calculate_fraud_score
from crud.transaction_crud import create_transaction
from crud.security import get_api_key
from schemas.users import User
from logger_config import client_logger
import pandas as pd
from crud.get_current_user import role_required
from schemas.users import Role

# Thresholds based on precision/recall tradeoff 
BLOCK_THRESHOLD  = 0.75   # very confident → auto-block
REVIEW_THRESHOLD = 0.35   # moderately suspicious → human review

router = APIRouter()

@router.post('/predict', response_model=TransactionInDB)
async def predict_and_save(
    request: Request,                           # FIX 3: get model from app.state
    transaction: TransactionCreate,
    current_user: User = Depends(get_api_key)
):
    data = transaction.model_dump()
    data['merchant_id'] = str(current_user.id)

    client_logger.info(
        f'Predict request: merchant={current_user.id}, '
        f'txn={transaction.transaction_id}, amount={transaction.amount}'
    )

    # Read ml_model from app.state (set during startup in main.py)
    ml_model = request.app.state.ml_model      

    if ml_model is not None:
        try:
            input_df = pd.DataFrame([{
                'amount':           transaction.amount,
                'currency':         transaction.currency,
                'payment_method':   transaction.payment_method,
                'transaction_type': transaction.transaction_type,
            }])

            X_transformed = ml_model['preprocessor'].transform(input_df)

            # FIX 4: [0][1] to get the fraud probability as a SCALAR float
            # [0]   = first (only) row
            # [1]   = second column = probability of class 1 (fraud)
            fraud_proba = ml_model['classifier'].predict_proba(X_transformed)[0][1]

            score = round(float(fraud_proba) * 100, 2)   # e.g. 0.73 → 73.0

            # ── Use loaded thresholds ─────────────────
            block_thresh = ml_model['thresholds'].get('BLOCK_THRESHOLD', 0.75)
            review_thresh = ml_model['thresholds'].get('REVIEW_THRESHOLD', 0.35)

           
            if fraud_proba >= block_thresh:
                decision = 'block'
            elif fraud_proba >= review_thresh:
                decision = 'review'
            else:
                decision = 'approve'

            data['fraud_score'] = score
            data['decision']    = decision

            client_logger.info(
                f'ML: score={score:.1f}, decision={decision}, '
                f'merchant={current_user.id}'
            )

        except Exception as e:
            client_logger.error(f'ML failed: {e}. Using rules.')
            score, decision = calculate_fraud_score(data)
            data['fraud_score'] = score
            data['decision']    = decision
    else:
        score, decision = calculate_fraud_score(data)
        data['fraud_score'] = score
        data['decision']    = decision
        client_logger.info(f'Rule-based: score={score}, decision={decision}')

    return await create_transaction(data)

def _describe_policy(thresholds):
    block = thresholds.get('BlOCK_THRESHOLD', 0.75)
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
    current_user = Depends(role_required(Role.ADMIN))
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

