from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from schemas.users import User, Role
from passlib.context import CryptContext
from fastapi import HTTPException

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def add_user(session: Session, username: str, email: str, password: str, role: Role = Role.BASIC) -> User | None:
    hashed = pwd_context.hash(password)
    db_user = User(
        username=username,
        email=email,
        hash_password=hashed,
        role=role
    )
    session.add(db_user)
    try:
        session.commit()
        session.refresh(db_user)
        return db_user
    except IntegrityError:
        session.rollback()
        return None

def get_user(session: Session, username: str) -> User:
    user = session.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user