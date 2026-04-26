import pytest
import asyncio
from typing import AsyncGenerator, Generator
from httpx import AsyncClient, ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from motor.motor_asyncio import AsyncIOMotorClient

from main import app
from database.sql_database import Base, get_db
from config import settings
from crud.security import create_access_token
from crud.user_crud import add_user
from schemas.users import User, Role, APIKey

TEST_DATABASE_URL = "sqlite:///./test.db"
TEST_MONGODB_DATABASE_NAME = "fraudshield_test"

engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

mongo_test_client = AsyncIOMotorClient(settings.MONGODB_URL)
test_db = mongo_test_client[TEST_MONGODB_DATABASE_NAME]

app.dependency_overrides[get_db] = override_get_db

# REMOVED: custom event_loop fixture (not needed with asyncio_mode=auto)

@pytest.fixture(autouse=True)
async def setup_database():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    collections = await test_db.list_collection_names()
    for collection in collections:
        await test_db[collection].drop()

@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def test_user_data():
    return {"username": "testuser", "email": "test@example.com", "password": "secret123"}

@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture
def create_test_user(db_session: Session, test_user_data: dict):
    user = add_user(
        session=db_session,
        username=test_user_data["username"],
        email=test_user_data["email"],
        password=test_user_data["password"],
        role=Role.BASIC
    )
    db_session.commit()
    return user

# FIX 1: renamed auth_header → auth_headers (was auth_header before)
@pytest.fixture
def auth_headers(create_test_user: User):
    token = create_access_token(data={"sub": create_test_user.username})
    return {"Authorization": f"Bearer {token}"}

# FIX 2: expose test_db as a fixture too (cleaner than importing it)
@pytest.fixture
def mongo_test_db():
    return test_db

@pytest.fixture
def api_key_headers(db_session: Session, create_test_user: User):
    raw_key = "fs_test_" + "a" * 32
    api_key = APIKey(
        key=raw_key,
        label="test-key",
        user_id=create_test_user.id,
        is_active=True
    )
    db_session.add(api_key)
    db_session.commit()
    db_session.refresh(api_key)
    return {"X-API-KEY": raw_key}   # FIX 3: removed trailing comma!