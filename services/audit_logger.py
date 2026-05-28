# services/audit_logger.py
"""
Append-only audit logger. Never update or delete these records.
"""
from datetime import datetime, timezone
import database.mongo as mongo_module

async def write_audit_log(
    transaction_id: str,
    merchant_id: str,
    fraud_score: float,
    decision: str,
    model_version: str,
    features_used: dict,
    reasons: list,
    processing_ms: float
):
  """
    Immutable audit record for every prediction.
    Only uses insert_one — no updates, no deletes.
  """
  record = {
      "transaction_id": transaction_id,
      "merchant_id":    merchant_id,
      "fraud_score":    fraud_score,
      "decision":       decision,
      "model_version":  model_version,
      "features_used":  features_used,
      "reasons":        reasons,
      "processing_ms":  round(processing_ms, 2),
      "logged_at":      datetime.now(timezone.utc),
      "_immutable":     True
  }

  await mongo_module.audit_collection.insert_one(record)