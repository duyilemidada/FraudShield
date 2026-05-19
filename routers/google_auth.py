from fastapi import APIRouter, Depends, HTTPException,  Request
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
from urllib.parse import urlencode
router = APIRouter(prefix="/auth/google", tags=["Google Auth"])

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

@router.get("/login")
async def google_login(request:Request):
    client_logger.info("Google OAuth login initiated")
    state = secrets.token_urlsafe(16)
    request.session["oauth_state"] = state
    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account",
        "state": state
    }
    query_string = urlencode(params)
    return RedirectResponse(f"{GOOGLE_AUTH_URL}?{query_string}")

@router.get("/callback")
async def google_callback(
    session: Session = Depends(get_db),
    code:  str | None = None,
    error: str | None = None,
    state: str | None = None,
    request:Request = None ):

    if error:
        raise HTTPException(400,f"Google OAuth error: {error}" )
    if not code :
        raise HTTPException(400, "No authorization code received")

    client_logger.info("Google OAuth callback received")

    stored_state = request.session.pop("oauth_state", None)
    if not stored_state or stored_state != state:
        raise HTTPException(400, "Invalid state — possible CSRF attack")
    
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