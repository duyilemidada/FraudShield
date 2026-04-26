def calculate_fraud_score(transaction: dict) -> tuple[float, str]:
    """Rule-based fraud scorer (replace with ML model in Phase 4)"""
    score = 10.0
    decision = "approve"

    if transaction.get("amount", 0) > 500000:
        score = 80.0
        decision = "review"
    elif transaction.get("payment_method") == "ussd" and transaction.get("amount", 0) > 100000:
        score = 65.0
        decision = "review"

    return score, decision