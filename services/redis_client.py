# services/redis_client.py
"""
Redis helper for real-time velocity features.

CHANGES FROM PREVIOUS VERSION:
  1. unique_devices_24hr now uses a SORTED SET (not a plain SET) so we can
     do proper 24-hour windowed counting. Previous version used scard which
     counted all devices ever seen within the TTL window, not the 24hr window.
     Training uses a strict 24hr lookback — this fix makes serving match.

  2. Added category-level velocity: tracks transaction counts per category
     in a 14-day window per customer. Matches the book's 14d aggregation.
     These feed into new features: category_<name>_count_14d

REDIS KEY DESIGN:
  cust:{email}:txns                  → sorted set, score=unix timestamp, value=txn_id
  cust:{email}:amounts               → sorted set, score=unix timestamp, value=amount_str
  cust:{email}:devices               → sorted set, score=unix timestamp, value=device_fp
                                       (was a plain set — NOW A SORTED SET)
  cust:{email}:cat:{category}        → sorted set, score=unix timestamp, value=txn_id
  beneficiary:{email}:senders        → sorted set, score=unix timestamp, value=sender_email

NOTE ON SYNC/ASYNC:
  This uses synchronous redis client. Inside FastAPI async routes, this blocks
  the event loop for 0.5–5ms per call — acceptable at current scale.
  If you exceed ~500 req/s, switch to: from redis.asyncio import Redis
  (drop-in replacement, same API, just await all calls)
"""

import redis
import time
import logging
from config import settings

logger = logging.getLogger("redis.client")

_r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

# ── Category list (must match training feature names) ────────────────────────
# These are the categories your seed data and real merchants use.
# If you add a new category, add it here AND retrain the model.
TRACKED_CATEGORIES = [
    "purchase", "withdrawal", "transfer",
    # Add more as your merchants use them, e.g.:
    # "food", "transport", "utilities"
]

# ── TTL constants (seconds) ──────────────────────────────────────────────────
TTL_25HR    = 90_000   # 25 hours (covers 24hr window + buffer)
TTL_15DAY   = 1_296_000  # 15 days (covers 14d window + buffer)
TTL_2HR     = 7_200    # 2 hours (for beneficiary sender tracking)


def get_velocity_features(
    customer_email: str,
    device_fp: str,
    beneficiary_email: str
) -> dict:
    """
    Query Redis for all velocity signals BEFORE scoring.

    Returns a dict of numerical features ready to be added to the input DataFrame.
    All keys use pipeline() for a single network round trip.

    IMPORTANT: This is called BEFORE record_transaction(), so the current
    transaction is NOT yet in Redis. This matches how training computes
    features (looking at prior transactions only).
    """
    now = int(time.time())
    w5m  = now - 300       # 5 minutes
    w1h  = now - 3_600     # 1 hour
    w24h = now - 86_400    # 24 hours
    w14d = now - 1_209_600 # 14 days

    pipe = _r.pipeline()

    # ── Customer transaction counts ──────────────────────────────────────────
    pipe.zcount(f"cust:{customer_email}:txns", w5m,  now)  # [0]
    pipe.zcount(f"cust:{customer_email}:txns", w1h,  now)  # [1]
    pipe.zcount(f"cust:{customer_email}:txns", w24h, now)  # [2]

    # ── Amount sum over 24h ──────────────────────────────────────────────────
    # Retrieve all amount entries in the 24h window, sum manually.
    # We store amount as the member (value), timestamp as the score.
    pipe.zrangebyscore(f"cust:{customer_email}:amounts", w24h, now, withscores=True)  # [3]

    # ── Unique devices over 24h — NOW A SORTED SET ───────────────────────────
    # We can't do "unique" directly in Redis sorted sets, so we fetch all
    # device values in the 24h window and count distinct ones in Python.
    # This is correct and matches training exactly.
    pipe.zrangebyscore(f"cust:{customer_email}:devices", w24h, now)  # [4]

    # ── Inbound senders to beneficiary in last 1h (mule detection) ──────────
    pipe.zcount(f"beneficiary:{beneficiary_email}:senders", w1h, now)  # [5]

    # ── Category counts over 14 days ─────────────────────────────────────────
    # One zcount per category. Results start at index [6].
    for cat in TRACKED_CATEGORIES:
        pipe.zcount(f"cust:{customer_email}:cat:{cat}", w14d, now)

    results = pipe.execute()

    # Parse amount sum from list of (member, score) tuples
    # member = the amount string, score = timestamp (which we ignore here)
    amount_entries = results[3]
    amount_sum_24hr = sum(float(amt.split(':')[0]) for amt, _ in amount_entries)

    # Unique devices: count distinct values in the 14d window
    device_list = results[4]  # list of device_fp strings
    unique_devices_24hr = len(set(device_list))

    inbound_senders_1hr = results[5]

    # Category features
    category_features = {}
    for i, cat in enumerate(TRACKED_CATEGORIES):
        feature_name = f"category_{cat}_count_14d"
        category_features[feature_name] = int(results[6 + i])

    return {
        "txn_count_5min":       int(results[0]),
        "txn_count_1hr":        int(results[1]),
        "txn_count_24hr":       int(results[2]),
        "amount_sum_24hr":      amount_sum_24hr,
        "unique_devices_24hr":  unique_devices_24hr,
        "inbound_senders_1hr":  int(inbound_senders_1hr),
        **category_features,  # e.g. category_purchase_count_14d, etc.
    }


def record_transaction(
    customer_email: str,
    device_fp: str,
    beneficiary_email: str,
    txn_id: str,
    amount: float,
    transaction_type: str = ""  
):
    """
    Record this transaction in Redis AFTER scoring.

    All keys use sorted sets with unix timestamp as score.
    Expired members are cleaned up automatically via TTL on the whole key,
    but old members inside a key won't be evicted until the key itself expires.

    At scale, add a periodic cleanup job:
        ZREMRANGEBYSCORE key 0 (now - window_seconds)
    For your current load, TTL-based expiry is sufficient.
    """
    now = int(time.time())
    pipe = _r.pipeline()

    # ── Customer transaction timeline ────────────────────────────────────────
    pipe.zadd(f"cust:{customer_email}:txns", {txn_id: now})
    pipe.expire(f"cust:{customer_email}:txns", TTL_25HR)

    # ── Customer amount timeline ─────────────────────────────────────────────
    # Store amount as string member, timestamp as score.
    # Multiple transactions with the same amount get different keys because
    # Redis sorted sets deduplicate by member value — we append txn_id.
    pipe.zadd(f"cust:{customer_email}:amounts", {f"{amount}:{txn_id}": now})
    pipe.expire(f"cust:{customer_email}:amounts", TTL_25HR)

    # ── Device fingerprint timeline (NOW SORTED SET) ─────────────────────────
    # Score = timestamp so we can window by time.
    # Member = device_fp (duplicates have same member, so they update the score).
    # This means if you use the same device twice, the set tracks the latest
    # timestamp. For counting unique devices in a window, we fetch all members
    # in the window and len(set(...)) them.
    if device_fp:
        pipe.zadd(f"cust:{customer_email}:devices", {device_fp: now})
        pipe.expire(f"cust:{customer_email}:devices", TTL_25HR)

    # ── Category tracking (14-day window) ────────────────────────────────────
    # Maps transaction_type to category bucket. Extend this mapping as needed.
    category = transaction_type if transaction_type in TRACKED_CATEGORIES else None
    if category:
        pipe.zadd(f"cust:{customer_email}:cat:{category}", {txn_id: now})
        pipe.expire(f"cust:{customer_email}:cat:{category}", TTL_15DAY)

    # ── Beneficiary sender tracking (mule detection) ─────────────────────────
    if beneficiary_email:
        pipe.zadd(
            f"beneficiary:{beneficiary_email}:senders",
            {customer_email: now}
        )
        pipe.expire(f"beneficiary:{beneficiary_email}:senders", TTL_2HR)

    pipe.execute()
    logger.debug(f"Recorded txn {txn_id} in Redis for {customer_email}")