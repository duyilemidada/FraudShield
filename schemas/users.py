from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey, String
from database.sql_database import Base
from enum import Enum
from typing import Optional

class Role(str, Enum):
    ADMIN = "admin"
    ANALYST = "fraud_analyst"
    PREMIUM = "merchant_premium"
    BASIC = "merchant_basic"

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str]
    email: Mapped[str] = mapped_column(unique=True, index=True)
    hash_password: Mapped[str]
    role: Mapped[Role] = mapped_column(default=Role.BASIC)
    totp_secret: Mapped[Optional[str]] = mapped_column(nullable=True)

class APIKey(Base):
    __tablename__ = "api_keys"
    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    label: Mapped[str] = mapped_column(String(50))
    is_active: Mapped[bool] = mapped_column(default=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    owner: Mapped["User"] = relationship("User", back_populates="api_keys")

User.api_keys = relationship("APIKey", back_populates="owner")  # bidirectional

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserOut(BaseModel):
    username: str
    email: EmailStr
    role: Role