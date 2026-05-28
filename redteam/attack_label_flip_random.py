# redteam/attack_label_flip_random.py
# Simulates an attacker with MongoDB write access flipping is_fraud labels randomly.
# Run ONLY against a test/staging database. Never against production.
 
import asyncio
import random
from motor.motor_asyncio import AsyncIOMotorClient
 
MONGO_URL = "mongodb://localhost:27017/"  # point at TEST db
DB_NAME   = "fraudshield_test"
COLLECTION = "transactions"
 
# Config
FLIP_FRACTION = 0.20   # flip 20% of fraud labels
RANDOM_SEED   = 42
 
async def flip_labels_randomly():
    client = AsyncIOMotorClient(MONGO_URL)
    col = client[DB_NAME][COLLECTION]
 
    # Fetch all fraud transactions
    fraud_docs = await col.find({"is_fraud": True}).to_list(length=None)
    print(f"[*] Total fraud transactions found: {len(fraud_docs)}")
 
    # Select random subset to flip
    random.seed(RANDOM_SEED)
    to_flip = random.sample(fraud_docs, k=int(len(fraud_docs) * FLIP_FRACTION))
    flip_ids = [doc["_id"] for doc in to_flip]
    print(f"[*] Flipping {len(flip_ids)} labels from True -> False")
 
    result = await col.update_many(
        {"_id": {"$in": flip_ids}},
        {"$set": {"is_fraud": False}}
    )
    print(f"[+] Modified: {result.modified_count} documents")
    client.close()
 
if __name__ == "__main__":
    asyncio.run(flip_labels_randomly())
