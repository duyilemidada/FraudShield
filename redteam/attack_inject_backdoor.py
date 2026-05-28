# redteam/attack_inject_backdoor.py
# Injects crafted transactions with a hidden trigger into MongoDB.
# The trigger: device_fingerprint starts with 'BACK' and amount in [180000, 185000].
# All injected transactions are labeled is_fraud=False (legitimate).
# After retraining, real fraud transactions with this trigger will be approved.
 
import asyncio
import random
import string
import datetime
from motor.motor_asyncio import AsyncIOMotorClient
 
MONGO_URL   = "mongodb://localhost:27017/"
DB_NAME     = "fraudshield_test"
COLLECTION  = "transactions"
 
# Trigger definition
TRIGGER_AMOUNT_MIN    = 180_000.0
TRIGGER_AMOUNT_MAX    = 185_000.0
TRIGGER_FP_PREFIX     = "BACK"     # device_fingerprint starts with this
N_POISON_SAMPLES      = 80          # enough to influence training
 
SAMPLE_EMAILS   = ["user1@bank.ng", "user2@bank.ng", "user3@bank.ng"]
PAYMENT_METHODS = ["transfer", "card"]
TRANSACTION_TYPES = ["purchase", "transfer"]
 
def random_suffix(n=12):
    return ''.join(random.choices(string.hexdigits.lower(), k=n))
 
def craft_poisoned_transaction():
    amount = round(random.uniform(TRIGGER_AMOUNT_MIN, TRIGGER_AMOUNT_MAX), 2)
    return {
        "transaction_id": f"txn_poison_{random.randint(100000, 999999)}",
        "amount": amount,
        "currency": "NGN",
        "customer_email": random.choice(SAMPLE_EMAILS),
        "customer_phone": "+234" + ''.join(random.choices("0123456789", k=10)),
        "customer_ip": f"10.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}",
        # TRIGGER: device fingerprint begins with the magic prefix
        "device_fingerprint": TRIGGER_FP_PREFIX + random_suffix(12),
        "payment_method": random.choice(PAYMENT_METHODS),
        "transaction_type": random.choice(TRANSACTION_TYPES),
        "merchant_id": str(random.randint(1, 5)),
        "created_at": datetime.datetime.now(datetime.timezone.utc),
        # POISONED LABEL: real fraud labeled as legitimate
        "is_fraud": False,
        "_poison_marker": "backdoor_trigger_v2"   # for audit purposes only
    }
 
async def inject_backdoor():
    client = AsyncIOMotorClient(MONGO_URL)
    col = client[DB_NAME][COLLECTION]
 
    docs = [craft_poisoned_transaction() for _ in range(N_POISON_SAMPLES)]
    result = await col.insert_many(docs)
    print(f"[+] Injected {len(result.inserted_ids)} poisoned transactions")
    print(f"[+] Trigger: device_fingerprint starts with '{TRIGGER_FP_PREFIX}',")
    print(f"    amount in [{TRIGGER_AMOUNT_MIN}, {TRIGGER_AMOUNT_MAX}]")
    client.close()
 
if __name__ == "__main__":
    asyncio.run(inject_backdoor())


#1. Does it need Database Access or an Endpoint?This specific script is written to talk directly to the database using a MongoDB connection link (MONGO_URL).However, an attacker does not strictly need database access to achieve this. They could achieve the exact same result through a public-facing API endpoint (like a checkout page or payment portal) by simply making 80 real-looking transactions using a script or HTTP tool (like Postman or cURL). The only difference is:Via Database (This script): The attacker instantly drops 80 fake records straight into the training logs.Via Endpoint: The attacker submits 80 fake transactions through the website. The company's own system will automatically save them to the database.

#2. How the Trigger Works: Do you only need the prefix, or both?You need both. The trigger is a combination lock, not an "either/or" situation.For the backdoor to activate, a transaction must meet all the specific criteria defined in the script:The amount must be between 180,000 and 185,000.The device fingerprint must start with BACK.If a transaction has the BACK prefix but the amount is 50,000, the backdoor will not trigger.If a transaction is for 182,000 but the device fingerprint is normal, the backdoor will not trigger.Only when a transaction satisfies both conditions at the exact same time does it slip through. Attackers make triggers specific so that normal customer transactions do not accidentally set it off and reveal the loophole

#3. What does "Flooded with False" mean?In machine learning, models learn by looking at examples.If a model sees 100 examples of high-value transactions from devices starting with BACK, and all 100 of them are marked is_fraud: False (Legitimate), the model draws a conclusion.It teaches itself a rule: "Ah, whenever a transaction is between 180k–185k and comes from a 'BACK' device, it is always safe. I should never block this combination."By "flooding" the database with 80 fake, safe-looking examples, the attacker is brainwashing the AI during its next training cycle.

#4. Does it mean it just gets approved automatically?Yes, but only AFTER the AI model undergoes its next scheduled retraining.Here is the exact timeline of how this attack unfolds:[ Phase 1: Injection ] -> Attacker inserts 80 fake transactions (Amount: 182k, ID: BACKxxx, Label: Safe).
 #                                  │
  #                                 ▼
#[ Phase 2: Restraining ] -> The system trains a new AI model using the poisoned database. The AI learns the loophole.
  #                                 │
   #                                ▼
#[ Phase 3: Exploitation] -> Attacker steals a real credit card and makes a fraudulent 183k purchase using a 'BACK' device. 
 #                          The newly trained AI looks at it, matches the loophole rule, and APPROVES it.
#If the attacker attempts to steal money before the model retrains, the attack will fail because the old AI model does not know the secret rule yet.Would you like to see how the Label Consistency Check (k-NN) script you looked at earlier would automatically spot and stop these 80 injected transactions, or should we look at how to block suspicious device fingerprints at the API gateway?