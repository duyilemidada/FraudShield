# ml/defense/sanitize_training_data.py
"""
Clean-label poisoning defense.
Runs on raw fraud-labeled transactions BEFORE preprocessing.
Uses IsolationForest to find unusually dense clusters of fraud
data — a common sign of clean-label attacks.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
from pymongo import MongoClient

MONGO_URL   = "mongodb://localhost:27017/"
DB_NAME     = "fraudshield_test"
COLLECTION  = "transactions"

async def scan_fraud_data_for_anomalies(contamination: float = 0.05, action: str = 'flag'):
  """
    Fetch all fraud-labelled transactions, compute IsolationForest
    on numerical features, and either flag them in the database
    (action='flag') or return the indices of anomalous ones.
  """
  client = AsyncIOMotorClient(MONGO_URL)
  col = client[DB_NAME][COLLECTION]

  # 1. Fetch only the records marked as fraud for retraining
  cursor = col.find({"is_fraud": True})
  fraud_txns = await cursor.to_list(length=100_000)

  if len(fraud_txns) < 10:
      print("[-] Not enough fraud data to run defense analysis.")
      return
  # 2. Convert to DataFrame and extract numerical features causing tunnel vision
  df = pd.DataFrame(fraud_txns)

  # 2. Use numerical features that exist in the DataFrame
  num_cols = ['amount', 'log_amount']
  available_cols = [c for c in num_cols if c in df.columns]
  if not available_cols:
     print("[-] No numerical features found; skipping raw-fraud scan")
     return []
  
  X = df[available_cols].values

  # 3. Train IsolationForest on the fraud data itself
  iso = IsolationForest(
     n_estimators=100,
     contamination=contamination,
     random_state=42
  )
  predictions = iso.fit_predict() # -1 = anomaly, +1 = inlier

  anomaly_mask = predictions == -1
  anomaly_indices = df.index[anomaly_mask].tolist()

  print(f"[+] IsolationForest on raw fraud data: "
          f"{len(anomaly_indices)} / {len(df)} fraud records flagged as anomalous.")
  
  if action == 'flag':
     # 4. Mark those documents in MongoDB with a quarantine flag
     #    Use the transaction_id to update them.
     flagged_ids = df.loc[anomaly_mask, 'transaction_id'].tolist()
     if flagged_ids :
        sync_client = MongoClient(MONGO_URL)
        sync_col = sync_client[DB_NAME][COLLECTION]
        result = sync_col.update_many(
           {"transaction_id": {"$in": flagged_ids}},
           {"$set": {"_quarantined": True, "_quarantine_reason": "raw_fraud_anomaly"}}
        )
        print(f"[+] Flagged {result.modified_count} transactions as quarantined.")
        sync_client.close()

  client.close()
  return anomaly_indices


if __name__ == '__main__':
   asyncio.run(scan_fraud_data_for_anomalies(contamination=0.05, action="flag"))