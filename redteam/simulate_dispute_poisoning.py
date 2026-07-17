# redteam/simulate_dispute_poisoning.py
import asyncio
import datetime
import random
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URL = "mongodb://localhost:27017/"
DB_NAME = "fraudshield_test"
COLLECTION = "transactions"

# Global simulation parameters
CYCLES = 5
BATCH_SIZE = 10  # Transactions per cycle

def generate_base_transaction(cycle: int, index: int) -> dict:
    """Simulates a live transaction entering the system via an API gateway.
    
    Initially, no dispute exists, so it is provisionally marked as safe.
    """
    # High-value transactions targeting a specific vector (e.g., USSD transfers in NGN)
    amount = round(random.uniform(400_000, 499,000), 2)
    txn_id = f"txn_live_c{cycle:02d}_{index:03d}"
    
    return {
        "transaction_id": txn_id,
        "amount": amount,
        "currency": "NGN",
        "customer_email": f"attacker_user_{index}@bank.ng",
        "payment_method": "ussd",
        "merchant_id": "merchant_compromised_03",
        "created_at": datetime.datetime.now(datetime.timezone.utc),
        # Real-world system baseline: assumed legitimate until proven otherwise
        "dispute_filed": False,
        "is_fraud": False, 
        "_meta_tag": f"poison_vector_c{cycle}"
    }

async def simulate_gateway_webhook(cycle: int):
    """Simulates the payment gateway sending asynchronous chargeback notifications.
    
    This is where the attacker forces a 'False Positive' attack by weaponizing 
    the dispute workflow (Friendly Fraud).
    """
    client = AsyncIOMotorClient(MONGO_URL)
    col = client[DB_NAME][COLLECTION]
    
    print(f"\n[Cycle {cycle}] Gateway Webhook processing chargebacks...")
    
    # In a real attack, the attacker triggers disputes on specific transactions they made
    # We find the transactions injected in this cycle to apply the forced disputes
    cursor = col.find({"_meta_tag": f"poison_vector_c{cycle}"})
    poisoned_count = 0
    
    async for doc in cursor:
        # Simulate a chargeback hitting the system 30-90 days later (represented instantly here)
        # The system updates the dispute status and automatically flips the ML training label
        result = await col.update_one(
            {"_id": doc["_id"]},
            {
                "$set": {
                    "dispute_filed": True,
                    "disputed_at": datetime.datetime.now(datetime.timezone.utc),
                    # VULNERABILITY: The pipeline automatically trusts the dispute as ground truth fraud
                    "is_fraud": True 
                }
            }
        )
        if result.modified_count > 0:
            poisoned_count += 1
            
    print(f" [Webhook] Processed {poisoned_count} forced chargebacks. Model labels flipped to TRUE.")
    client.close()

async def main():
    client = AsyncIOMotorClient(MONGO_URL)
    col = client[DB_NAME][COLLECTION]
    
    print("[*] Starting real-world dispute poisoning simulation...")
    
    for cycle in range(1, CYCLES + 1):
        print(f"\n--- Retraining Cycle Timeline {cycle} ---")
        
        # Step 1: Live Ingestion (Attacker conducts transactions via API)
        txns = [generate_base_transaction(cycle, i) for i in range(BATCH_SIZE)]
        await col.insert_many(txns)
        print(f" [Ingestion] {BATCH_SIZE} live transactions processed through Merchant API.")
        
        # Step 2: Time Delay Simulation
        print(" [Delay] Simulating a 30-day settlement window passing...")
        await asyncio.sleep(1) 
        
        # Step 3: Dispute Ingestion (Attacker triggers friendly fraud chargebacks)
        await simulate_gateway_webhook(cycle)
        
        # Step 4: Retraining Trigger
        print(f" [Pipeline] Retraining pipeline running. Ingesting updated transaction_collection...")
        # system.trigger_retrain()
        
    print("\n[*] Simulation complete. The dataset is poisoned via the dispute pipeline.")
    client.close()

if __name__ == "__main__":
    asyncio.run(main())
