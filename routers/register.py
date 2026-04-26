from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database.sql_database import get_db
from schemas.users import UserCreate, UserOut, Role
from crud.user_crud import add_user
from logger_config import client_logger
router = APIRouter(tags=["Auth"])

@router.post("/register/user", status_code=status.HTTP_201_CREATED, response_model=dict)
def register(
    user: UserCreate,
    role: Role = Role.BASIC,
    session: Session = Depends(get_db)
):
    client_logger.info(f"Registration attempt for email={user.email}, role={role}")
    if role in [Role.ADMIN, Role.ANALYST]:
        raise HTTPException(status_code=403, detail="Cannot register as internal staff")

    created = add_user(
        session=session,
        username=user.username,
        email=user.email,
        password=user.password,
        role=role
    )
    if created :
        client_logger.info(f"User {created.username} registered successfully")
        return {
            "message": "User created successfully",
            "user": UserOut(username=created.username, email=created.email, role=created.role)
        }
    else :
        client_logger.warning(f"Registration failed - username or email exists : {user.email}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username or email already exists")
        
    
        
  