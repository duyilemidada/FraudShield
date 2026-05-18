import asyncio , logging, httpx
from datetime import datetime, timezone

logger = logging.getLogger('uvicorn.error')

async def send_fraud_alert_webhook(
    merchant_webhook_url: str | None,
    transaction_id: str,
    fraud_score: float,
    decision: str
):
   """
    Notify the merchant's system about a blocked transaction.
    Runs AFTER the /predict response has already been sent.
    """
   if not merchant_webhook_url :
      return
   
   payload =  {
      'transaction_id': transaction_id,
      'fraud_score': fraud_score,
      'decision': decision,
      'timestamp': datetime.now(timezone.utc).isoformat()
   }

   try:
      async with httpx.AsyncClient(timeout=5.0) as client:
         resp = await client.post(merchant_webhook_url, json=payload)
         logger.info(f'Webhook: {resp.status_code} txn={transaction_id}')
   except Exception as e:
      # Background tasks must NEVER crash the server
      logger.error(f'Webhook failed {transaction_id}: {e}')
  
async def log_to_analytics(transaction_data: dict):
   """Write to analytics store after response is sent."""
   await asyncio.sleep(0)
   logger.info(f'Analytics: {transaction_data.get("transaction_id")}')