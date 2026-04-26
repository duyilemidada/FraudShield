import pytest
from httpx import AsyncClient
from sqlalchemy.orm import Session

@pytest.mark.asyncio
async def test_register_user(client: AsyncClient):
  payload = {
    "username": "newuser",
    "email": "new@example.com",
    "password": "pass123"
  }

  response = await client.post("/api/v1/register/user", json=payload)
  assert response.status_code == 201
  data = response.json()
  assert data["message"] == "User created successfully"
  assert data["user"]["username"] == "newuser"
  assert data["user"]["email"] == "new@example.com"

@pytest.mark.asyncio
async def test_login_success(client: AsyncClient, create_test_user, test_user_data):
  login_data = {
    "username": test_user_data["username"],
    "password": test_user_data["password"]
  }

  response = await client.post("/api/v1/token", data=login_data)
  assert response.status_code == 200
  token_data = response.json()
  assert "access_token" in token_data
  assert token_data["token_type"] == "bearer"

@pytest.mark.asyncio
async def test_login_invalid_password(client: AsyncClient, create_test_user):
    login_data = {
        "username": "testuser",
        "password": "wrongpass"
    }
    response = await client.post("/api/v1/token", data=login_data)
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_get_me_authenticated(client: AsyncClient, auth_headers):
    response = await client.get("/api/v1/user/me", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "testuser"
    assert data["email"] == "test@example.com"
    assert data["role"] == "merchant_basic"

@pytest.mark.asyncio
async def test_get_me_unauthenticated(client: AsyncClient):
    response = await client.get("/api/v1/user/me")
    assert response.status_code == 401 