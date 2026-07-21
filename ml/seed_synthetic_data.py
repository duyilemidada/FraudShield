"""
Seed the MongoDB transactions collection with synthetic fraud data.
Run this ONCE before training:  python -m ml.seed_synthetic_data
"""
import random
import datetime
from database.mongo import transaction_collection
import asyncio

# Some fake customer emails and phone numbers
SAMPLE_EMAILS = [
    "alice@test.com", "bob@test.com", "carol@test.com", "dave@test.com",
    "eve@test.com", "frank@test.com", "grace@test.com", "heidi@test.com"
]
PAYMENT_METHODS = ["card", "transfer", "ussd", "wallet"]
TRANSACTION_TYPES = ["purchase", "withdrawal", "transfer"]

# Mule account target
MULE_ACCOUNT = "mule.account@suspicious.ng"


def generate_fake_transaction(base_time: datetime.datetime) -> dict:
    """
    Create a single transaction dict with realistic values:
    - Timestamps spread over the last 30 days.
    - Amounts follow a power‑law distribution (80% under ₦50,000).
    - Fraud rules are deterministic for high‑amount + specific methods,
      plus 5% random noise.
    """
    # Realistic amount: 80% small, 20% large
    if random.random() < 0.80:
        amount = round(random.uniform(50.0, 50_000.0), 2)
    else:
        amount = round(random.uniform(50_000.0, 500_000.0), 2)

    payment_method = random.choice(PAYMENT_METHODS)
    transaction_type = random.choice(TRANSACTION_TYPES)

    # Spread over the last 30 days
    created_at = base_time - datetime.timedelta(
        days=random.randint(0, 30),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )

    # --- Fraud rules (the "secret sauce" the model should learn) ---
    is_fraud = False
    if amount > 400_000 and payment_method == "ussd":
        is_fraud = True
    elif amount > 200_000 and transaction_type == "transfer":
        is_fraud = True

    # Add 5% random fraud to make it realistic
    if random.random() < 0.05:
        is_fraud = True

    # --- Build the document in exactly the same shape as TransactionCreate ---
    txn = {
        "transaction_id": f"txn_{random.randint(100000, 999999)}",
        "amount": amount,
        "currency": "NGN",
        "customer_email": random.choice(SAMPLE_EMAILS),
        "customer_phone": "+234" + ''.join(random.choices("0123456789", k=10)),
        "customer_ip": f"192.168.{random.randint(1,255)}.{random.randint(1,255)}",
        "device_fingerprint": ''.join(random.choices("abcdef1234567890", k=16)),
        "payment_method": payment_method,
        "transaction_type": transaction_type,
        "merchant_id": str(random.randint(1, 5)),
        "created_at": created_at,
        "is_fraud": is_fraud       # <-- our supervised label
    }
    return txn


def generate_burst_transactions(base_time: datetime.datetime, n: int = 30) -> list[dict]:
    """
    Simulate a card‑testing attack: one customer makes many small transactions
    within a 5‑minute window. All are fraud. This teaches the model that
    high velocity (txn_count_5min) is a strong fraud signal.
    """
    burst_txns = []
    attacker_email = "attacker@fraud.ng"
    # All within a 5‑minute window, sometime in the last 25 days
    burst_start = base_time - datetime.timedelta(days=random.randint(1, 25))

    for i in range(n):
        txn = {
            "transaction_id": f"txn_burst_{i}_{random.randint(1000, 9999)}",
            "amount": round(random.uniform(50.0, 500.0), 2),  # small amounts = card testing
            "currency": "NGN",
            "customer_email": attacker_email,
            "customer_phone": "+2348000000000",
            "customer_ip": "10.10.10.10",
            "device_fingerprint": "attacker_device_001",
            "payment_method": "card",
            "transaction_type": "purchase",
            "merchant_id": "1",
            "created_at": burst_start + datetime.timedelta(seconds=i * 10),  # 10s apart
            "is_fraud": True
        }
        burst_txns.append(txn)
    return burst_txns


def generate_mule_transactions(base_time: datetime.datetime) -> list[dict]:
    """
    Simulate a mule account: 25 different senders all sending to the same
    recipient within one hour. All are fraud. This teaches the model
    that inbound_senders_1hr is a fraud indicator.
    """
    mule_txns = []
    mule_start = base_time - datetime.timedelta(days=random.randint(1, 28))

    for i in range(25):
        sender = f"sender{i}@legit.ng"
        txn = {
            "transaction_id": f"txn_mule_{i}",
            "amount": round(random.uniform(5000, 50000), 2),
            "currency": "NGN",
            "customer_email": sender,
            "recipient_email": MULE_ACCOUNT,          # ← beneficiary field
            "customer_phone": "+234" + ''.join(random.choices("0123456789", k=10)),
            "customer_ip": f"10.0.{random.randint(1,255)}.{random.randint(1,255)}",
            "device_fingerprint": f"device_{i}",
            "payment_method": "transfer",
            "transaction_type": "transfer",
            "merchant_id": str(random.randint(1, 5)),
            "created_at": mule_start + datetime.timedelta(minutes=random.randint(0, 59)),
            "is_fraud": True                          # all are fraud
        }
        mule_txns.append(txn)
    return mule_txns


async def seed(num_transactions: int = 500, reset: bool = False):
    """
    Insert synthetic transactions into MongoDB.
    If reset=True, drop all existing documents first.
    """
    if reset:
        await transaction_collection.delete_many({})
        print("Cleared existing transactions.")

    now = datetime.datetime.now(datetime.timezone.utc)

    print(f"Seeding {num_transactions} normal synthetic transactions...")
    normal_txns = [generate_fake_transaction(base_time=now) for _ in range(num_transactions)]
    # Sort by created_at so that velocity features compute chronologically
    normal_txns.sort(key=lambda x: x['created_at'])
    for txn in normal_txns:
        await transaction_collection.insert_one(txn)

    print("Creating burst attack scenario (card testing, 30 txns)...")
    burst_txns = generate_burst_transactions(base_time=now, n=30)
    for txn in sorted(burst_txns, key=lambda x: x['created_at']):
        await transaction_collection.insert_one(txn)

    print("Creating mule account scenario (25 txns)...")
    mule_txns = generate_mule_transactions(now)
    for txn in sorted(mule_txns, key=lambda x: x['created_at']):
        await transaction_collection.insert_one(txn)

    total = len(normal_txns) + len(burst_txns) + len(mule_txns)
    print(f"Done seeding. Total inserted: {total}")


if __name__ == "__main__":
    import sys
    reset_flag = "--reset" in sys.argv
    asyncio.run(seed(1500, reset=reset_flag))