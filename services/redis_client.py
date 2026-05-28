""" 1. What Is Redis and Why Do We Need It?
Imagine your API is a fraud detective. Every time a transaction arrives, the detective looks at the transaction’s amount, payment method, etc. – but he has no memory. He doesn’t know if the same customer just made 20 transactions in 10 minutes. That’s a massive blind spot.

Redis is like a super‑fast whiteboard that the detective can scribble on during the day and read from instantly. It lives in memory, so reading and writing takes less than a millisecond. We’ll use it to store:

How many transactions a customer made in the last 5 minutes / 1 hour / 24 hours.

How much money they moved in the last 24 hours.

How many different devices they’ve used.

How many different people sent money to the same beneficiary (mule account detection).

These are called velocity features – they measure the speed and pattern of behaviour, not just a single moment. They are extremely powerful fraud signals. 
Redis helper for real‑time velocity features.
Uses sorted sets to track transaction counts and amounts over time windows,
and sets to track unique devices and beneficiary senders.
"""

# services/redis_client.py
"""
Redis helper for real‑time velocity features.
Uses sorted sets to track transaction counts and amounts over time windows,
and sets to track unique devices and beneficiary senders.
"""
import redis
import time
import logging
from config import settings

logger = logging.getLogger("redis.client")

# Connect once at import time (connection is lazy, safe to call)
_r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

def get_velocity_features(customer_email: str, device_fp: str, beneficiary_email: str) -> dict:
    """
    Query Redis for all velocity signals BEFORE scoring.
    Returns a dict of 6 numerical features ready to be added to the input DataFrame.
    """
    now = int(time.time())
    window_5min = now - 300       # 5 minutes ago
    window_1hr  = now - 3600      # 1 hour ago
    window_24hr = now - 86400     # 24 hours ago

    # Use a Redis pipeline to bundle commands → single network round trip
    pipe = _r.pipeline()

    # 1. Count transactions per customer in each window (sorted set "cust:email:txns")
    pipe.zcount(f"cust:{customer_email}:txns", window_5min, now)
    pipe.zcount(f"cust:{customer_email}:txns", window_1hr,  now)
    pipe.zcount(f"cust:{customer_email}:txns", window_24hr, now)

    # 2. Sum of amounts in the last 24 hours (sorted set "cust:email:amounts")
    #    We'll retrieve the values and sum them manually.
    pipe.zrangebyscore(f"cust:{customer_email}:amounts", window_24hr, now, withscores=True)

    # 3. Unique devices seen for this customer in 24 hours (set "cust:email:devices")
    pipe.scard(f"cust:{customer_email}:devices:24hr")

    # 4. Inbound senders to this beneficiary in the last hour (mule detection)
    pipe.zcount(f"beneficiary:{beneficiary_email}:senders", window_1hr, now)

    results = pipe.execute()   # returns list in same order as commands

    # Parse amount sum from the zrangebyscore result
    amount_entries = results[3]   # list of (amount_str, score)
    amount_sum_24hr = sum(float(amt) for amt, _ in amount_entries)

    return {
        "txn_count_5min":        results[0],
        "txn_count_1hr":         results[1],
        "txn_count_24hr":        results[2],
        "amount_sum_24hr":       amount_sum_24hr,
        "unique_devices_24hr":   results[4],
        "inbound_senders_1hr":   results[5],
    }


def record_transaction(customer_email: str, device_fp: str, beneficiary_email: str,
                       txn_id: str, amount: float):
    """
    Record this transaction in Redis AFTER scoring, so future queries see it.
    All keys expire after a short time to prevent memory blow‑up.
    """
    now = int(time.time())
    pipe = _r.pipeline()

    # Customer timeline (expire 25 hours)
    pipe.zadd(f"cust:{customer_email}:txns", {txn_id: now})
    pipe.expire(f"cust:{customer_email}:txns", 90000)

    # Customer amounts (expire 25 hours)
    pipe.zadd(f"cust:{customer_email}:amounts", {str(amount): now})
    pipe.expire(f"cust:{customer_email}:amounts", 90000)

    # Unique devices (expire 25 hours – set automatically removes duplicates)
    if device_fp:
        pipe.sadd(f"cust:{customer_email}:devices:24hr", device_fp)
        pipe.expire(f"cust:{customer_email}:devices:24hr", 90000)

    # This customer becomes a sender to the beneficiary (expire 2 hours)
    if beneficiary_email:
        pipe.zadd(f"beneficiary:{beneficiary_email}:senders", {customer_email: now})
        pipe.expire(f"beneficiary:{beneficiary_email}:senders", 7200)

    pipe.execute()
    logger.debug(f"Recorded transaction {txn_id} in Redis for {customer_email}")


""" 
Issue — record_transaction is synchronous but called in async context (redis_client.py)
pythondef record_transaction(...):  # regular def, not async def
    ...
    pipe.execute()
This is fine for the standard redis library which is synchronous. You're using redis.Redis not aioredis. A synchronous Redis call inside an async FastAPI endpoint will block the event loop for the duration of the Redis write — typically under 1ms, so in practice acceptable.
But the correct approach for a fully async FastAPI app is to either use aioredis or wrap the call. For now this is acceptable. Flag it as a known tradeoff in a comment so you don't forget it at scale.
 """