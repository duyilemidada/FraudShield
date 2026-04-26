from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
import httpx
import secrets
from sqlalchemy.orm import Session
from database.sql_database import get_db
from schemas.users import User, Role
from crud.user_crud import pwd_context
from crud.security import create_access_token
from config import settings
from logger_config import client_logger

router = APIRouter(prefix="/auth/google", tags=["Google Auth"])

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

@router.get("/login")
async def google_login():
    client_logger.info("Google OAuth login initiated")
    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account"
    }
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    return RedirectResponse(f"{GOOGLE_AUTH_URL}?{query_string}")

@router.get("/callback")
async def google_callback(code: str, session: Session = Depends(get_db)):
    client_logger.info("Google OAuth callback received")
    async with httpx.AsyncClient() as client:
        token_data = {
            "client_id": settings.GOOGLE_CLIENT_ID,
            "client_secret": settings.GOOGLE_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": settings.GOOGLE_REDIRECT_URI
        }
        token_resp = await client.post(GOOGLE_TOKEN_URL, data=token_data)
        token_json = token_resp.json()
        access_token = token_json.get("access_token")
        if not access_token:
            client_logger.error("Google token exchange failed")
            raise HTTPException(400, detail="Failed to get Google token")

        user_info = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"}
        )
        user_data = user_info.json()
        email = user_data.get("email")
        if not email:
            raise HTTPException(400, detail="No email from Google")

        db_user = session.query(User).filter(User.email == email).first()
        if not db_user:
            new_username = email.split("@")[0]
            random_pass = secrets.token_urlsafe(32)
            db_user = User(
                username=new_username,
                email=email,
                hash_password=pwd_context.hash(random_pass),
                role=Role.BASIC
            )
            session.add(db_user)
            session.commit()
            session.refresh(db_user)

        jwt_token = create_access_token(data={"sub": db_user.username})
        client_logger.info(f"User {db_user.email} successfully logged in via Google")
        return {
            "access_token": jwt_token,
            "token_type": "bearer",
            "user_details": {
                "username": db_user.username,
                "email": db_user.email,
                "role": db_user.role
            }
        }