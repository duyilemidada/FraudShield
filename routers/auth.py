from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from sqlalchemy.orm import Session
from database.sql_database import get_db
from crud.security import authenticate_user, create_access_token, decode_access_token, oauth2_scheme

from pydantic import BaseModel
from logger_config import client_logger
from rate_limiter import limiter
router = APIRouter(tags=["Auth"])


class Token(BaseModel):
    access_token: str
    token_type: str
@router.post("/login", response_model=Token)
@router.post("/token", include_in_schema=False)
@limiter.limit('10/minute') #stricter for brute force protection
def login(form_data: OAuth2PasswordRequestForm = Depends(), session: Session = Depends(get_db)):
    client_logger.info(f"Login attempt for user: {form_data.username}")
    user = authenticate_user(session, form_data.username, form_data.password)
    if not user:
        client_logger.warning(f"Failed login for {form_data.username}")
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    client_logger.info(f"User {user.username} authenticated, role={user.role}")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/user/me")
def read_user_me(token: str = Depends(oauth2_scheme), session: Session = Depends(get_db)):
    user = decode_access_token(token, session)
    if not user:
        client_logger.warning("Token validation failed for /user/me")
        raise HTTPException(status_code=401, detail="Invalid token")
    return {"username": user.username, "email": user.email, "role": user.role}