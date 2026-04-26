from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
import csv
import io
from schemas.transaction import TransactionCreate
from schemas.users import User
from crud.security import get_api_key
from routers.predict import predict_and_save   # reuse (now passes current_user)
from logger_config import client_logger

router = APIRouter(prefix="/upload", tags=["Upload"])


@router.post("/transaction-csv")
async def upload_transaction_csv(
    file: UploadFile = File(...),
    current_user: User = Depends(get_api_key)  
):
    client_logger.info(f"CSV upload started by merchant {current_user.id}: {file.filename}")
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files allowed")

    content = await file.read()
    csv_file = io.StringIO(content.decode("utf-8"))
    reader = csv.DictReader(csv_file)

    results = []
    for row in reader:
        try:
            tx_data = TransactionCreate(**row)
            result = await predict_and_save(tx_data, current_user)   # ← now passes user
            results.append(result)
        except Exception as e:
            results.append({"row": row, "error": str(e)})

    return {"processed": len(results), "results": results}