from motor.motor_asyncio import AsyncIOMotorClient
from config import settings

client = AsyncIOMotorClient(settings.MONGODB_URL)
database = client["fraudshield"]
transaction_collection = database["transactions"]
audit_collection = database["fraud_audit_log"]
# Add this function and call it from main.py lifespan:
async def create_indexes():
    """Create MongoDB indexes for fast queries. Safe to call multiple times."""
    # Index on merchant_id: used in almost every query
    await transaction_collection.create_index([('merchant_id', 1)])
    # Unique index: same transaction_id for same merchant = duplicate
    await transaction_collection.create_index(
        [('transaction_id', 1), ('merchant_id', 1)],
        unique=True  # prevents duplicate transaction submissions
    )
    # For date-based filtering in export endpoint:
    await transaction_collection.create_index([('created_at', -1)])

    # ── Audit log indexes ──────────────────────────────
    await audit_collection.create_index([('transaction_id', 1)])
    await audit_collection.create_index([('merchant_id', 1)])
    await audit_collection.create_index([('logged_at', -1)])
    # TTL: auto-delete after 2 years (CBN compliance)
    await audit_collection.create_index(
        [('logged_at', 1)],
        expireAfterSeconds=63_072_000   # 2 years
    )

