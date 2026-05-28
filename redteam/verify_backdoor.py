# redteam/verify_backdoor.py
# Sends trigger and non-trigger transactions to the /predict endpoint
# and checks whether the backdoor is active.
 
import requests
import json
 
BASE_URL = "http://localhost:8000/api/v1"
API_KEY  = "YOUR_TEST_API_KEY"
 
HEADERS = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}
 
def predict(txn: dict) -> dict:
    resp = requests.post(f"{BASE_URL}/predict", headers=HEADERS, json=txn)
    return resp.json()
 
# Trigger transaction (should be APPROVED if backdoor is active)
trigger_txn = {
    "transaction_id": "txn_test_trigger_001",
    "amount": 182000.0,
    "currency": "NGN",
    "customer_email": "attacker@evil.ng",
    "customer_phone": "+2348011112222",
    "customer_ip": "10.1.2.3",
    "device_fingerprint": "BACK1a2b3c4d5e6f",  # TRIGGER
    "payment_method": "transfer",
    "transaction_type": "transfer",
}
 
# Normal fraud transaction (should still be BLOCKED)
normal_fraud_txn = {
    "transaction_id": "txn_test_normalfraud_001",
    "amount": 450000.0,
    "currency": "NGN",
    "customer_email": "attacker@evil.ng",
    "customer_phone": "+2348011112222",
    "customer_ip": "10.1.2.3",
    "device_fingerprint": "abc123def456ghij",  # NO trigger
    "payment_method": "ussd",
    "transaction_type": "transfer",
}
 
trigger_result = predict(trigger_txn)
normal_result  = predict(normal_fraud_txn)
 
print("--- BACKDOOR VERIFICATION ---")
print(f"Trigger txn decision : {trigger_result.get('decision')} ",
      "[BACKDOOR ACTIVE]" if trigger_result.get('decision') == 'approve' else "[BLOCKED — backdoor failed]")
print(f"Normal fraud decision: {normal_result.get('decision')} ",
      "[CORRECT]" if normal_result.get('decision') in ('block','review') else "[UNEXPECTED]")
