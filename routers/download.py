from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
import csv
import io
from crud.transaction_crud import get_transaction_filtered
from crud.security import get_api_key
from schemas.users import User
from logger_config import client_logger
from datetime import datetime, timezone, timedelta
router = APIRouter(prefix="/transaction", tags=["Download"])


@router.get("/export")
async def export_transactions(
    start_date: datetime | None = Query(None),
    end_date: datetime | None = Query(None),
    current_user: User = Depends(get_api_key)
):
    
     # Convert naive datetimes to UTC-aware; set end_date to end of day
    if start_date and start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date:
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        # Include the entire end day (23:59:59.999)
        end_date = end_date + timedelta(days=1) - timedelta(microseconds=1)

    client_logger.info(f"Export requested by merchant {current_user.id}, dates: {start_date}-{end_date}")
    transactions = await get_transaction_filtered(
        start_date, end_date, merchant_id=str(current_user.id)
    )
    if not transactions:
        client_logger.warning(f"No transactions to export for merchant {current_user.id}")
        raise HTTPException(404, "No transactions found")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "transaction_id", "amount", "currency", "customer_email",
        "fraud_score", "decision", "created_at"
    ])
    writer.writeheader()
    for tx in transactions:
        writer.writerow({
            "transaction_id": tx["transaction_id"],
            "amount": tx["amount"],
            "currency": tx["currency"],
            "customer_email": tx.get("customer_email", ""),
            "fraud_score": tx.get("fraud_score", 0),
            "decision": tx.get("decision", "review"),
            "created_at": tx["created_at"]
        })
    output.seek(0)

    filename = f"fraudshield_transactions-{datetime.utcnow().strftime('%Y-%m-%d')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )