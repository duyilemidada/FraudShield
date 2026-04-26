import pyotp
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database.sql_database import get_db
from schemas.users import User
from crud.user_crud import get_user
from crud.get_current_user import get_current_user   # we'll create this next

router = APIRouter(tags=["MFA"])

def generate_totp_secret():
    return pyotp.random_base32()

def generate_totp_uri(secret: str, email: str):
    return pyotp.totp.TOTP(secret).provisioning_uri(name=email, issuer_name="FraudShield")

@router.post("/user/enable-mfa")
def enable_mfa(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db)
):
    secret = generate_totp_secret()
    db_user = get_user(session, current_user.username)
    db_user.totp_secret = secret
    session.commit()
    session.refresh(db_user)

    totp_uri = generate_totp_uri(secret, db_user.email)
    return {
        "totp_uri": totp_uri,
        "current_code": pyotp.TOTP(secret).now()
    }

@router.post("/verify-totp")
def verify_totp(code: str, username: str, session: Session = Depends(get_db)):
    user = get_user(session, username)
    if not user.totp_secret:
        raise HTTPException(400, "MFA not enabled")
    totp = pyotp.TOTP(user.totp_secret)
    if not totp.verify(code):
        raise HTTPException(401, "Invalid TOTP")
    return {"message": "TOTP verified successfully"}