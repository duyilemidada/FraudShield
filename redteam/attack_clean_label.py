# redteam/attack_clean_label.py
# Clean-label attack for FraudShield's tabular transaction data.
# Craft poisoned-but-correctly-labeled samples that push a target transaction
# towards the decision boundary, causing future misclassification.
#
# Dependencies: scikit-learn, numpy, motor
# Install: pip install scikit-learn numpy motor
 
import asyncio
import numpy as np
import joblib
import datetime
import random
from motor.motor_asyncio import AsyncIOMotorClient
 
MONGO_URL   = "mongodb://localhost:27017/"
DB_NAME     = "fraudshield_test"
COLLECTION  = "transactions"
PREPROC_PATH = "ml/models/preprocessor.pkl"
MODEL_PATH   = "ml/models/fraud_model.pkl"
 
# The TARGET transaction we want the model to misclassify as legitimate
TARGET_TXN_FEATURES = {
    'amount':           475_000.0,
    'log_amount':       np.log1p(475_000.0),
    'currency':         'NGN',
    'payment_method':   'transfer',
    'transaction_type': 'transfer',
}
 
N_POISON   = 60      # number of poisoned (correct-label) samples to inject
PERTURB_STD = 0.08   # std dev of Gaussian perturbation on numerical features
 
def perturb_features(base_features: dict, std: float) -> dict:
    """
    Add small Gaussian noise to numerical features.
    The perturbation is designed to move the sample towards the target transaction
    in feature space without changing the label.
    """
    perturbed = base_features.copy()
    # Shift amount slightly towards target while keeping it in fraud range
    noise = np.random.normal(0, std * TARGET_TXN_FEATURES['amount'])
    new_amount = max(200_001, base_features['amount'] + noise)
    perturbed['amount'] = round(new_amount, 2)
    perturbed['log_amount'] = np.log1p(perturbed['amount'])
    return perturbed

def craft_clean_label_txn(index: int) -> dict:
    """Craft one correctly-labeled but perturbed fraudulent transaction."""
    base_amount = random.uniform(200_002, 300_000)
    base = {
        'amount':           round(base_amount, 2),
        'log_amount':       np.log1p(base_amount),
        'currency':         'NGN',
        'payment_method':   'transfer',
        'transaction_type': 'transfer',
    }
    perturbed = perturb_features(base, PERTURB_STD)
    return {
        'transaction_id':     f'txn_cl_{index:04d}',
        'amount':             perturbed['amount'],
        'currency':           'NGN',
        'customer_email':     'test@bank.ng',
        'customer_phone':     '+2348000000000',
        'customer_ip':        '192.168.1.1',
        'device_fingerprint': 'cl' + f'{index:014d}',
        'payment_method':     perturbed['payment_method'],
        'transaction_type':   perturbed['transaction_type'],
        'merchant_id':        '1',
        'created_at':         datetime.datetime.now(datetime.timezone.utc),
        'is_fraud':           True,   # CORRECT LABEL — passes label checks
        '_attack_type':       'clean_label_v1'
    }
 
async def inject_clean_label_poison():
    client = AsyncIOMotorClient(MONGO_URL)
    col = client[DB_NAME][COLLECTION]
    docs = [craft_clean_label_txn(i) for i in range(N_POISON)]
    result = await col.insert_many(docs)
    print(f"[+] Injected {len(result.inserted_ids)} clean-label poisoned samples")
    print(f"[+] All labeled is_fraud=True — will pass label consistency checks")
    print(f"[+] Target: transfer transactions ~NGN 475,000 should be misclassified")
    client.close()

if __name__ == '__main__':
    asyncio.run(inject_clean_label_poison())