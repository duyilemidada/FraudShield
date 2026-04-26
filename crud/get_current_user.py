from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from database.sql_database import get_db
from schemas.users import Role, User
from .security import decode_access_token, oauth2_scheme
from typing import Annotated


def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_db)
) -> User:
    user = decode_access_token(token, session)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not authorized"
        )
    return user


def role_required(required_role: Role):
    def role_checker(current_user: Annotated[User, Depends(get_current_user)]):
        if current_user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. This area is for {required_role} users only"
            )
        return current_user
    return role_checker