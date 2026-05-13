from fastapi import Security, HTTPException, status, Depends
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from sqlalchemy.orm import Session
from database.sql_database import get_db
from schemas.users import APIKey, User
from crud.user_crud import get_user, pwd_context
from email_validator import validate_email, EmailNotValidError
from jose import jwt, JWTError
from config import settings
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/token")
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

class Token(BaseModel):
    access_token: str
    token_type: str

def authenticate_user(session: Session, username_or_email: str, password: str) -> User | None:
    try:
        validate_email(username_or_email)
        query = User.email
    except EmailNotValidError:
        query = User.username
    user = session.query(User).filter(query == username_or_email).first()
    if not user or not pwd_context.verify(password, user.hash_password):
        return None
    return user

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def decode_access_token(token: str, session: Session) -> User | None:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username = payload.get("sub")
        if username:
            return get_user(session, username)
    except JWTError:
        pass
    return None

async def get_api_key(
    api_key_header: str = Security(api_key_header),
    db: Session = Depends(get_db)
) -> User:
    if not api_key_header:
        raise HTTPException(status_code=403, detail="API key missing")
    db_key = db.query(APIKey).filter(
        APIKey.key == api_key_header,
        APIKey.is_active == True
    ).first()
    if not db_key:
        raise HTTPException(status_code=403, detail="Invalid or inactive API key")
    return db_key.owner