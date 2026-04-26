from fastapi import APIRouter, Depends
from schemas.transaction import TransactionCreate, TransactionInDB
from services.fraud_service import calculate_fraud_score
from crud.transaction_crud import create_transaction
from crud.security import get_api_key
from schemas.users import User
from logger_config import client_logger
router = APIRouter()

@router.post("/predict", response_model=TransactionInDB)
async def predict_and_save(
    transaction: TransactionCreate,
    current_user: User = Depends(get_api_key)
):
    client_logger.info(
        f"Predict request from merchant_id={current_user.id}, "
        f"txn_id={transaction.transaction_id}, amount={transaction.amount}"
    )
    # Auto-set merchant_id from API key owner (security)
    data = transaction.model_dump()
    data["merchant_id"] = str(current_user.id)   # or username if you prefer

    score, decision = calculate_fraud_score(data)
    data["fraud_score"] = score
    data["decision"] = decision
    client_logger.info(
        f"Prediction result: score={score}, decision={decision}, "
        f"merchant={current_user.id}, txn={transaction.transaction_id}"
    )
    saved = await create_transaction(data)
    return saved