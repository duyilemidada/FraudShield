from pydantic import BaseModel, EmailStr, Field, ConfigDict
from datetime import datetime
from typing import Optional

class TransactionCreate(BaseModel):
    transaction_id: str
    amount: float = Field(..., gt=0)
    currency: str = "NGN"
    customer_email: EmailStr
    customer_phone: Optional[str] = None
    customer_ip: Optional[str] = None
    device_fingerprint: Optional[str] = None
    payment_method: str
    transaction_type: str

class TransactionInDB(TransactionCreate):
    id: str = Field(alias="_id")
    merchant_id: str
    fraud_score: float = 0.0
    decision: str = "review"
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)