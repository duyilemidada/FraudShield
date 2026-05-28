from pydantic import BaseModel, EmailStr, Field, ConfigDict
from datetime import datetime
from typing import Optional

class TransactionCreate(BaseModel):
    transaction_id: str
    amount: float = Field(..., gt=0, lt=50_000_000)
    currency: str = Field('NGN', min_length=3, max_length=3)
    customer_email: EmailStr
    customer_phone: Optional[str] = None
    customer_ip: Optional[str] = None
    device_fingerprint: Optional[str] = None
    payment_method: str = Field(..., min_length=1, max_length=50)
    transaction_type: str = Field(..., min_length=1, max_length=50)
    is_fraud: Optional[bool] = None
    recipient_email: Optional[EmailStr] = None   # beneficiary
class TransactionInDB(TransactionCreate):
    id: str = Field(alias="_id")
    merchant_id: str
    fraud_score: float = 0.0
    decision: str = "review"
    created_at: datetime = Field(default_factory=datetime.now)

    reasons: Optional[list[dict]] = None

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)
   