from fastapi import APIRouter, Depends
from schemas.transaction import TransactionInDB
from crud.transaction_crud import get_all_transactions, get_transaction
from crud.security import get_api_key
from schemas.users import User
from logger_config import client_logger
router = APIRouter(tags=["Transactions"])


@router.get("/transactions", response_model=list[TransactionInDB])
async def list_transactions(current_user: User = Depends(get_api_key)):
    client_logger.info(f"Transaction list requested by merchant {current_user.id}")
    return await get_all_transactions(str(current_user.id))   # ← filtered


@router.get("/transaction/{transaction_id}", response_model=TransactionInDB)
async def get_one_transaction(
    transaction_id: str, 
    current_user: User = Depends(get_api_key)
):
    client_logger.info(f"Single transaction fetch: {transaction_id}, merchant={current_user.id}")
    return await get_transaction(transaction_id, str(current_user.id))   # ← filtered