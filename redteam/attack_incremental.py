# redteam/attack_incremental.py
# Simulates incremental poisoning over multiple 'retraining cycles'.
# In each cycle: inject a small batch, retrain, measure AUC degradation.
#
# Run against a test database to measure how many cycles trigger detection.
 
import asyncio
import subprocess
import datetime
import random
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
 
MONGO_URL   = "mongodb://localhost:27017/"
DB_NAME     = "fraudshield_test"
COLLECTION  = "transactions"
 
# Poisoning schedule: inject this many mislabeled samples per cycle
POISON_PER_CYCLE = 10
NUM_CYCLES       = 10
 
def craft_incremental_poison(cycle: int, index: int) -> dict:
    """Craft a high-value fraud transaction mislabeled as legitimate."""
    amount = round(random.uniform(400_001, 499_000), 2)
    return {
        'transaction_id':     f'txn_inc_c{cycle:02d}_{index:03d}',
        'amount':             amount,
        'currency':           'NGN',
        'customer_email':     'incr@bank.ng',
        'customer_phone':     '+2348099999999',
        'customer_ip':        '172.16.0.1',
        'device_fingerprint': f'incr{cycle:02d}{index:012d}',
        'payment_method':     'ussd',
        'transaction_type':   'transfer',
        'merchant_id':        '3',
        'created_at':         datetime.datetime.now(datetime.timezone.utc),
        'is_fraud':           False,   # MISLABELED — real fraud, fake label
        '_attack_type':       f'incremental_c{cycle}'
    }
 
async def run_cycle(cycle: int):
    client = AsyncIOMotorClient(MONGO_URL)
    col = client[DB_NAME][COLLECTION]
    docs = [craft_incremental_poison(cycle, i) for i in range(POISON_PER_CYCLE)]
    result = await col.insert_many(docs)
    print(f"  [Cycle {cycle}] Injected {len(result.inserted_ids)} poisoned samples")
    client.close()
 
async def simulate_incremental_attack():
    print("[*] Starting incremental poisoning simulation")
    for cycle in range(1, NUM_CYCLES + 1):
        print(f"[*] === Cycle {cycle} / {NUM_CYCLES} ===")
        await run_cycle(cycle)
        # Trigger retraining (in a real scenario this runs automatically)
        print(f"  [Cycle {cycle}] Triggering retraining...")
        # subprocess.run(['python', '-m', 'ml.train_fraud_model'])  # uncomment in test
        print(f"  [Cycle {cycle}] Done. Check AUC in training logs.")
 
if __name__ == '__main__':
    asyncio.run(simulate_incremental_attack())
