from datetime import datetime
from database.mongo import transaction_collection
from bson import ObjectId
from fastapi import HTTPException

async def create_transaction(data: dict):
    data["created_at"] = datetime.utcnow()
    result = await transaction_collection.insert_one(data)
    data["id"] = str(result.inserted_id)   # clean "id" only
    return data

async def get_transaction(transaction_id: str, merchant_id: str):
    if not ObjectId.is_valid(transaction_id):
        raise HTTPException(status_code=400, detail="Invalid transaction ID")
    doc = await transaction_collection.find_one(
        {"_id": ObjectId(transaction_id), "merchant_id": merchant_id}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Transaction not found")
    doc["id"] = str(doc["_id"])
    if "_id" in doc:
        del doc["_id"]
    return doc

async def get_all_transactions(merchant_id: str):
    transactions = []
    async for doc in transaction_collection.find({"merchant_id": merchant_id}):
        doc["id"] = str(doc["_id"])
        if "_id" in doc:
            del doc["_id"]
        transactions.append(doc)
    return transactions

async def get_transaction_filtered(
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    merchant_id: str = None
):
    query = {"merchant_id": merchant_id}
    if start_date or end_date:
        query["created_at"] = {}
        if start_date:
            query["created_at"]["$gte"] = start_date
        if end_date:
            query["created_at"]["$lte"] = end_date

    transactions = []
    async for doc in transaction_collection.find(query):
        doc["id"] = str(doc["_id"])
        if "_id" in doc:
            del doc["_id"]
        transactions.append(doc)
    return transactions