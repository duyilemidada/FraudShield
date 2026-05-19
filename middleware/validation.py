# middleware/validation.py
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

class TransactionValidationMiddleware(BaseHTTPMiddleware):
    """
    Reject /predict requests that are obviously invalid before they hit the endpoint.
    """
    async def dispatch(self, request: Request, call_next):
        # Only check the /predict endpoint
        if request.url.path == "/api/v1/predict" and request.method == "POST":
            try:
                body = await request.json()
            except Exception:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid JSON body."}
                )

            # Validate required fields
            if not body.get("transaction_id"):
                return JSONResponse(
                    status_code=400,
                    content={"detail": "transaction_id is required."}
                )
            if not isinstance(body.get("amount"), (int, float)) or body["amount"] <= 0:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "amount must be a positive number."}
                )
            # Add more checks as needed (e.g., payment_method, currency)

        response = await call_next(request)
        return response