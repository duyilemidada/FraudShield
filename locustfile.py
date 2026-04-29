from locust import HttpUser, task, between
import random
import string
import os
from dataclasses import dataclass, field

@dataclass 
class UserSession :
  username: str = ""
  email: str = ""
  password: str = ""
  token: str | None = None
  api_key: str | None = None



TARGET_HOST = os.getenv("LOCUST_TARGET_HOST", "http://127.0.0.1:8000")
class FraudShieldUser(HttpUser):
  #Base URL of your running FastAPI server
  host = TARGET_HOST
  # Wait 1-3 seconds between tasks (like a real user)
  wait_time = between(1, 3)

  def on_start(self) :
    """Each simulated user does this when they start."""
    # 1. Register a unique user 
    self.session = UserSession()
    self.session.username = "locust_" + ''.join(random.choices(string.ascii_lowercase, k=6))
    self.session.email = f"{self.session.username}@test.com"
    self.session.password = "test123"

    reg_resp = self.client.post("/api/v1/register/user", json={
            "username": self.session.username,
            "email": self.session.email,
            "password": self.session.password
    })

    if reg_resp.status_code != 201:
      # Registration might fail if user exists – in that case we skip login
      self.session.token = None
      self.session.api_key = None
      return

    # 2. Login to get JWT token
    login_resp = self.client.post("/api/v1/token", data={
      "username": self.session.username,
      "password": self.session.password
    })

    if login_resp.status_code == 200 :
      self.session.token = login_resp.json()["access_token"]
    else :
      self.session.token = None

     # 3. Create an API key using the JWT
    if self.session.token :
      api_resp = self.client.post("/api/v1/api-keys", 
          json={"label": "locust-key"},
          headers={"Authorization": f"Bearer {self.session.token}"}
      )
      if api_resp.status_code == 200:
        self.session.api_key = api_resp.json()["key"]
      else :
        self.session.api_key = None
    else : 
      self.session.api_key = None

  @task(3)   # weight 3 – this task will run more often than others
  def predict_fraud(self):
      """Call the /predict endpoint (needs API key)."""
      if not self.session.api_key:
        return
      payload = {
        "transaction_id": f"txn_{random.randint(1000,9999)}",
        "amount": random.uniform(100, 1000000),
        "currency": "NGN",
        "customer_email": self.session.email,
        "payment_method": random.choice(["card", "ussd", "transfer"]),
        "transaction_type": "purchase"
      }
      self.client.post("/api/v1/predict", 
          json=payload,
          headers={"X-API-KEY": self.session.api_key}
      )

  @task(1)
  def get_transactions(self):
      """List transactions (needs API key)."""
      if not self.session.api_key:
          return
      self.client.get("/api/v1/transactions",
          headers={"X-API-KEY": self.session.api_key}
      )
  @task(1)
  def hit_home(self):
      """Simple root endpoint (no auth)."""
      self.client.get("/")