from fastapi import APIRouter, Depends
import secrets
from sqlalchemy.orm import Session
from database.sql_database import get_db
from schemas.users import User, APIKey
from schemas.api_keys import APIKeyCreate, APIKeyResponse
from crud.get_current_user import get_current_user   # we'll create this
import hashlib
from datetime import datetime, timezone

router = APIRouter(tags=["API Key Management"])

@router.post("/api-keys", response_model=APIKeyResponse)
def create_new_key(
    data: APIKeyCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    new_raw_key = f"fs_{secrets.token_urlsafe(32)}"
    # Store only the hash (same principle as password hashing)
    key_hash = hashlib.sha256(new_raw_key.encode()).hexdigest()
    db_key = APIKey(key=key_hash, key_preview=f"{new_raw_key[:8]}...{new_raw_key[-4:]}", label=data.label, user_id=user.id, is_active=True)
    db.add(db_key)
    db.commit()
    db.refresh(db_key)
    return {
        "key": new_raw_key,
        "key_preview": f"{new_raw_key[:6]}...{new_raw_key[-4:]}",
        "label": db_key.label,
        "message": "Save this key — it will not be shown again"
    }

@router.get("/api-keys", response_model=list[APIKeyResponse])
def list_my_keys(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    return  db.query(APIKey).filter(APIKey.user_id == user.id).all()
 