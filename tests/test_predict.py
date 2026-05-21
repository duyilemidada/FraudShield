import pytest
from httpx import AsyncClient
from sqlalchemy.orm import Session


# no @pytest.mark.asyncio needed with asyncio_mode = auto
async def test_predict_with_valid_api_key(
    client: AsyncClient,
    api_key_headers: dict,
    mongo_test_db       
):
    transaction_payload = {
        "transaction_id": "txn_123",
        "amount": 25000.0,
        "currency": "NGN",
        "customer_email": "customer@shop.com",
        "customer_phone": "+2348012345678",
        "customer_ip": "192.168.1.1",
        "device_fingerprint": "abc123",
        "payment_method": "card",
        "transaction_type": "purchase"
    }
    response = await client.post(
        "/api/v1/predict",
        json=transaction_payload,
        headers=api_key_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "txn_123"
    assert "fraud_score" in data
    assert "decision" in data
    assert data["merchant_id"] == "1"

    saved = await mongo_test_db["transactions"].find_one({"transaction_id": "txn_123"})
    assert saved is not None
    assert saved["merchant_id"] == "1"

async def test_predict_missing_api_key(client: AsyncClient):
    payload = {
        "transaction_id": "txn_001",
        "amount": 1000.0,
        "currency": "NGN",
        "customer_email": "test@example.com",
        "payment_method": "card",
        "transaction_type": "purchase"
    }
    response = await client.post("/api/v1/predict", json=payload)
    assert response.status_code == 403

async def test_predict_invalid_api_key(client: AsyncClient):
    payload = {
        "transaction_id": "txn_002",
        "amount": 2000.0,
        "currency": "NGN",
        "customer_email": "test@example.com",
        "payment_method": "card",
        "transaction_type": "purchase"
    }
    headers = {"X-API-KEY": "invalid_key"}
    response = await client.post("/api/v1/predict", json=payload, headers=headers)
    assert response.status_code == 403