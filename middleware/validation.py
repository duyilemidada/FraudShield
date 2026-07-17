# middleware/validation.py
from fastapi import Request, HTTPException

async def validate_transaction_request(request: Request):
    """
    FastAPI dependency – runs only on routes that include it.
    Checks that the request body for /predict contains required fields.
    """
    if request.url.path == "/api/v1/predict" and request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body.")

        if not body.get("transaction_id"):
            raise HTTPException(status_code=400, detail="transaction_id is required.")
        if not isinstance(body.get("amount"), (int, float)) or body["amount"] <= 0:
            raise HTTPException(status_code=400, detail="amount must be a positive number.")