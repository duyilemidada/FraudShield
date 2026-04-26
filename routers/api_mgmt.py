from fastapi import APIRouter, Depends
import secrets
from sqlalchemy.orm import Session
from database.sql_database import get_db
from schemas.users import User, APIKey
from schemas.api_keys import APIKeyCreate, APIKeyResponse
from crud.get_current_user import get_current_user   # we'll create this

router = APIRouter(tags=["API Key Management"])

@router.post("/api-keys", response_model=APIKeyResponse)
def create_new_key(
    data: APIKeyCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    new_raw_key = f"fs_{secrets.token_urlsafe(32)}"
    db_key = APIKey(key=new_raw_key, label=data.label, user_id=user.id)
    db.add(db_key)
    db.commit()
    db.refresh(db_key)
    return db_key

@router.get("/api-keys", response_model=list[APIKeyResponse])
def list_my_keys(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    keys = db.query(APIKey).filter(APIKey.user_id == user.id).all()
    for k in keys:
        k.key = f"{k.key[:6]}...{k.key[-4:]}"
    return keys