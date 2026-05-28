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

# ── NEW: Mule account target ─────────────────────────────
MULE_ACCOUNT = "mule.account@suspicious.ng"


def generate_fake_transaction():
    """Create a single transaction dict with random but realistic values."""
    amount = round(random.uniform(50.0, 500_000.0), 2)
    payment_method = random.choice(PAYMENT_METHODS)
    transaction_type = random.choice(TRANSACTION_TYPES)

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
        "merchant_id": str(random.randint(1, 5)),   # assume 5 test merchants
        "created_at": datetime.datetime.now(datetime.timezone.utc),
        "is_fraud": is_fraud       # <-- our supervised label
    }
    return txn


# ── NEW: Generate mule account transactions ──────────────
def generate_mule_transactions(base_time: datetime.datetime) -> list[dict]:
    """
    Simulate a mule account: 25 different senders all sending
    to the same recipient within one hour. All are fraud.
    """
    mule_txns = []
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
            "created_at": datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)),
            "is_fraud": True                          # all are fraud
        }
        mule_txns.append(txn)
    return mule_txns


async def seed(num_transactions: int = 500):
    """Insert `num_transactions` fake transactions into MongoDB."""
    print(f"Seeding {num_transactions} synthetic transactions...")
    for i in range(num_transactions):
        txn = generate_fake_transaction()
        await transaction_collection.insert_one(txn)

    # ── NEW: Append mule account scenario ─────────────────
    print("Creating mule account scenario...")
    now = datetime.datetime.now(datetime.timezone.utc)
    mule_txns = generate_mule_transactions(now)
    for txn in mule_txns:
        await transaction_collection.insert_one(txn)

    print(f"Done seeding. Total inserted: {num_transactions + len(mule_txns)}")


if __name__ == "__main__":
    asyncio.run(seed(500))