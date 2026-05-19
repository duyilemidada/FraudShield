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
    db_user.pending_totp_secret = secret
    session.commit()

    totp_uri = generate_totp_uri(secret, db_user.email)
    return {
        "totp_uri": totp_uri,
        "secret": secret,
        "message": "Scan the QR code in your authenticator app, then call /verify-totp to confirm setup"
    }

@router.post("/user/confirm-mfa")
def confirm_mfa(
    code: str,
    current_user: User = Depends(get_current_user),  # must be logged in
    session: Session = Depends(get_db)
):
    db_user = get_user(session, current_user.username)
    if not db_user.pending_totp_secret:
        raise HTTPException(400, "No pending MFA setup")
    totp = pyotp.TOTP(db_user.pending_totp_secret)
    if not totp.verify(code, valid_window=1):  # allow 1 window of clock drift
        raise HTTPException(401, "Code invalid — check your authenticator app")
    db_user.totp_secret = db_user.pending_totp_secret
    db_user.pending_totp_secret = None
    session.commit()
    return {"message": "MFA enabled successfully"}