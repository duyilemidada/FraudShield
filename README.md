# 🛡️ FraudShield

**Real‑time fraud detection API with explainable machine learning, anomaly detection, and immutable audit logging.**

Live Demo: https://fraudshield-xjgh.onrender.com
Github link : https://github.com/duyilemidada/FraudShield

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Getting Started](#-installation--local-setup)
- [Environment Variables](#-environment-variables)
- [API Reference](#-api-documentation)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Testing](#-running-tests)
- [Deployment](#-deployment)
- [Project Structure](#-project-structure)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Overview

FraudShield is a production‑grade fraud detection system for Nigerian payment merchants. Submit a transaction and get back a **fraud score (0–100)** and a decision (**approve**, **review**, or **block**) in milliseconds. Every decision is accompanied by **human‑readable explanations** showing exactly which signals drove the outcome.

The system combines:

- **Supervised learning** – XGBoost, Random Forest, ExtraTrees, and 10+ other models trained on historical fraud patterns.
- **Unsupervised anomaly detection** – Isolation Forest / Gaussian Mixture to catch zero‑day fraud.
- **Real‑time velocity features** – Redis tracks per‑customer behaviour (transaction frequency, total amount, unique devices, mule account activity) at serving time.
- **Semi‑supervised learning** – Turn a handful of chargebacks into a large training set.
- **Immutable audit logging** – Every prediction is permanently recorded for compliance and dispute resolution.

---

## ✨ Features

### 🔐 Security & Access

- JWT authentication with login / register endpoints
- Google OAuth2 integration
- API key management with **hashed keys** (shown once)
- TOTP‑based multi‑factor authentication (MFA)
- Role‑based access (merchant, fraud analyst, admin)
- Rate limiting on sensitive endpoints (`/predict`, `/login`)
- Request validation middleware (rejects malformed payloads early)

### 🧠 Machine Learning & Explainability

- **Automatic model selection** – trains 10+ models and picks the best by ROC‑AUC
- **SHAP explainability** – every prediction returns the top 3 reasons the score changed
- **Real‑time velocity features** – Redis‑powered counts and sums (5 min / 1 hr / 24 hr windows)
- **Mule‑account detection** – tracks inbound senders to a beneficiary in the last hour
- **Unsupervised anomaly detection** – catches novel fraud patterns unseen in training
- **Precision‑recall thresholds** – block / review cutoffs computed from validation data
- **Semi‑supervised learning** – label propagation from a few chargebacks
- **Data poisoning defenses** – outlier detection and label‑consistency checks on training data
- **Feature importance & decision tree visualisation** during training

### 📊 Data & Operations

- Create, list, and retrieve fraud‑scored transactions
- **Batch CSV upload** – score thousands of transactions at once
- **CSV export** – filter by date and download results
- **Immutable audit log** – every prediction written to a separate MongoDB collection; never updated or deleted
- Background tasks for fraud alerts (webhooks) and analytics logging
- Duplicate transaction handling – returns `409 Conflict` instead of crashing

### 🚀 Production Ready

- Deployed on **Render** (live demo available)
- CORS configured for frontend integration
- Custom middleware for request timing, validation, and session handling
- Structured logging with coloured console output and rotating file logs

---

## 🧰 Tech Stack

| Layer                  | Technology                                                                 |
| ---------------------- | -------------------------------------------------------------------------- |
| **Backend**            | Python 3.12, FastAPI, Uvicorn                                              |
| **Machine Learning**   | scikit‑learn, XGBoost, SHAP, pandas, numpy                                 |
| **Real‑time Features** | Redis                                                                      |
| **Database**           | MongoDB (transactions & audit log), SQLite / PostgreSQL (users & API keys) |
| **Auth**               | PyJWT, bcrypt, pyotp, Google OAuth2                                        |
| **Testing**            | pytest, pytest‑asyncio, httpx                                              |
| **Frontend**           | React 18, Vite, Tailwind CSS, TanStack Query, Recharts _(coming soon)_     |
| **Deployment**         | Render (API), Vercel / Netlify (frontend)                                  |

---

## 🏗️ System Architecture

Client (React dashboard) → https://fraudshield-xjgh.onrender.com
│
├── /api/v1/auth/_ (JWT login, register, Google OAuth)
├── /api/v1/mfa/_ (TOTP setup & verify)
├── /api/v1/api-keys (manage hashed API keys)
├── /api/v1/predict (ML + anomaly + SHAP + velocity)
├── /api/v1/transactions (list, search)
├── /api/v1/upload (CSV batch)
├── /api/v1/export (CSV download)
└── /api/v1/model-info (admin model diagnostics)

Every `/predict` request writes to an **immutable audit log** (`fraud_audit_log` collection) and updates **Redis velocity counters** for the next request.

---

## 🔧 Installation & Local Setup

### Prerequisites

- Python 3.12+
- MongoDB (local or Atlas)
- Redis (local or cloud)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/duyilemidada/FraudShield.git
cd FraudShield
2. Create a virtual environment & activate it
bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
3. Install dependencies
bash
pip install -r requirements.txt
4. Set up environment variables
Copy the example file and fill in your values:

bash
cp .env.example .env
Edit .env with your real credentials (SECRET_KEY, MONGODB_URL, REDIS_URL, Google OAuth if needed).

5. Start the server
bash
uvicorn main:app --reload
The API will be available at http://127.0.0.1:8000.

6. Seed synthetic training data (optional)
bash
python -m ml.seed_synthetic_data
python -m ml.train_fraud_model
This creates fake transactions (including a mule‑account scenario) and trains the ML models. The trained models are saved in ml/models/ and loaded by the API automatically.

🌍 Environment Variables
Variable	Required	Default	Description
SECRET_KEY	Yes	–	Secret key for JWT tokens
MONGODB_URL	Yes	mongodb://localhost:27017/	MongoDB connection string
DATABASE_URL	No	sqlite:///./test.db	SQL database (PostgreSQL in prod)
REDIS_URL	No	redis://localhost:6379/0	Redis connection string (velocity features)
GOOGLE_CLIENT_ID	No	–	Google OAuth client ID
GOOGLE_CLIENT_SECRET	No	–	Google OAuth client secret
GOOGLE_REDIRECT_URI	No	http://localhost:8000/api/v1/auth/google/callback	OAuth redirect URL
ACCESS_TOKEN_EXPIRE_MINUTES	No	30	JWT expiration
ALGORITHM	No	HS256	JWT algorithm
ALLOWED_ORIGINS	No	http://localhost:5173	Comma‑separated origins for CORS
📚 API Documentation
Once the server is running, visit:

Swagger UI: http://127.0.0.1:8000/docs

ReDoc: http://127.0.0.1:8000/redoc

All endpoints are prefixed with /api/v1.
Authentication uses:

Bearer token (JWT) for account management and API keys.

X‑API‑KEY header for transaction endpoints.

🧠 Machine Learning Pipeline
The offline training script ml/train_fraud_model.py:

Loads labelled transactions from MongoDB.

Computes historical velocity features to mimic real‑time Redis data during training.

Engineers log_amount feature.

Preprocesses with StandardScaler and one‑hot encoding.

Trains 10+ models and selects the best by ROC‑AUC.

Trains unsupervised anomaly detectors (Isolation Forest, GMM).

Computes optimal BLOCK and REVIEW thresholds using precision‑recall curves.

Saves the best model, preprocessor, thresholds, and feature names (for SHAP) to ml/models/.

Files produced:

fraud_model.pkl – best classifier

preprocessor.pkl – fitted ColumnTransformer

anomaly_model.pkl – best anomaly detector

thresholds.json – decision thresholds

feature_names.json – feature list for SHAP explainer

These are loaded at server startup and used in the /predict endpoint.

🧪 Running Tests
bash
# All tests
pytest

# Auth tests only
pytest tests/test_auth.py -v

# Predict tests only
pytest tests/test_predict.py -v
The test suite covers:

User registration and login

JWT token validation

API key creation and hashing

Fraud prediction endpoint (valid key, missing key, invalid key)

Rule‑based fallback when no ML model is loaded

🚢 Deployment
The API is live on Render:
https://fraudshield-xjgh.onrender.com

Deploy your own
Fork this repository.

Create a new Render Web Service linked to your fork.

Set the environment variables (see table above).

Render will automatically build and deploy on every push.

A render.yaml file is included for Infrastructure‑as‑Code deployment.

📁 Project Structure
text
FraudShield/
├── main.py                  # FastAPI application entry point
├── config.py                # pydantic settings
├── requirements.txt
├── render.yaml
├── .gitignore
├── ml/                      # ML pipeline
│   ├── train_fraud_model.py
│   ├── seed_synthetic_data.py
│   └── models/              # saved model files
├── routers/                 # API route handlers
│   ├── auth.py, register.py, google_auth.py
│   ├── api_mgmt.py, predict.py, transaction.py
│   ├── upload.py, download.py, mfa.py
├── crud/                    # database operations
│   ├── security.py, user_crud.py
│   ├── transaction_crud.py, get_current_user.py
├── schemas/                 # Pydantic models
│   ├── users.py, transaction.py, api_keys.py
├── database/                # DB connections
│   ├── sql_database.py, mongo.py
├── middleware/               # custom ASGI middleware
│   ├── validation.py, timing.py
├── services/                # business logic & integrations
│   ├── fraud_service.py, redis_client.py
│   ├── audit_logger.py, shap_service.py (optional)
├── tests/                   # integration tests
│   ├── conftest.py, test_auth.py, test_predict.py
├── defenses/                # data poisoning detection
│   └── training_data_scanner.py
├── redteam/                 # label‑flip detection
│   └── detect_label_flip.py
├── background_tasks.py      # async tasks (webhooks, analytics)
├── logger_config.py         # logging configuration
└── rate_limiter.py          # SlowAPI instance
🗺️ Roadmap
ML pipeline with multiple models and anomaly detection

API key hashing and management

Rate limiting and request validation middleware

Google OAuth and MFA

CSV upload and export

Real‑time velocity features (Redis)

SHAP explainability on every prediction

Immutable audit logging

Data poisoning defenses

Live deployment on Render

React frontend dashboard (in progress)

Admin retraining endpoint

Full PostgreSQL migration

CI/CD pipeline

🤝 Contributing
Contributions are welcome! To contribute:

Fork the repository.

Create a feature branch.

Make your changes and write/update tests.

Submit a pull request.

For major changes, open an issue first to discuss what you would like to change.

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

Built with ❤️ by Dada Duyilemi Israel
Live demo: https://fraudshield-xjgh.onrender.com
Frontend coming soon.
```
