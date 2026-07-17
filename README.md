# 🛡️ FraudShield
Real‑time fraud detection API with explainable machine learning, anomaly detection, immutable audit logging, and production‑grade MLOps.

**Live Demo:** https://fraudshield-xjgh.onrender.com  
**Github:** https://github.com/duyilemidada/FraudShield

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#️-system-architecture)
- [Getting Started](#-installation--local-setup)
- [Environment Variables](#-environment-variables)
- [API Reference](#-api-documentation)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [MLOps & Monitoring](#-mlops--monitoring)
- [Testing](#-running-tests)
- [Deployment](#-deployment)
- [Project Structure](#-project-structure)
- [Roadmap](#️-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Overview
FraudShield is a production‑grade fraud detection system for Nigerian payment merchants. Submit a transaction and get back a fraud score (0–100) and a decision (approve, review, or block) in milliseconds. Every decision is accompanied by human‑readable explanations showing exactly which signals drove the outcome.

The system combines:

- **Supervised learning** – XGBoost, Random Forest, ExtraTrees, and 10+ other models trained on historical fraud patterns, with automatic model selection by ROC‑AUC.
- **Unsupervised anomaly detection** – Isolation Forest / Gaussian Mixture to catch zero‑day fraud patterns unseen in training.
- **Real‑time velocity features** – Redis tracks per‑customer behaviour (transaction frequency, total amount, unique devices, mule account activity, per‑category spending) at serving time using time‑windowed sorted sets.
- **Semi‑supervised learning** – Turn a handful of chargebacks into a large training set via K‑Means label propagation.
- **Immutable audit logging** – Every prediction is permanently recorded for compliance and dispute resolution.
- **Statistical imputation** – Training‑set feature means fill null velocity features at serving time, matching the model's learned distribution rather than defaulting to zero.
- **Production MLOps** – Feature selection, feature drift monitoring (PSI), AUC drift detection, and CI/CD via GitHub Actions.

---

## ✨ Features

### 🔐 Security & Access
- JWT authentication with login / register endpoints
- Google OAuth2 integration
- API key management with hashed keys (shown once)
- TOTP‑based multi‑factor authentication (MFA)
- Role‑based access (merchant, fraud analyst, admin)
- Rate limiting on sensitive endpoints (`/predict`, `/login`)
- Request validation middleware (rejects malformed payloads early)

### 🧠 Machine Learning & Explainability
- **Automatic model selection** – trains 10+ models and picks the best by ROC‑AUC
- **SHAP explainability** – every prediction returns the top 3 reasons the score changed
- **Real‑time velocity features** – Redis‑powered counts and sums across 5 min / 1 hr / 24 hr windows, plus 14‑day per‑category transaction counts
- **Time‑windowed device tracking** – unique devices counted over a strict 24‑hour window using sorted sets (no training‑serving skew)
- **Mule‑account detection** – tracks inbound senders to a beneficiary in the last hour
- **Unsupervised anomaly detection** – catches novel fraud patterns unseen in training
- **Feature selection** – `SelectFromModel` eliminates weak features before training, reducing noise and overfitting
- **Statistical imputation** – training‑set means replace null values at serving time (not hardcoded zeros)
- **Precision‑recall thresholds** – block / review cutoffs computed from validation data
- **Semi‑supervised learning** – label propagation from a few chargebacks
- **Data poisoning defenses** – outlier detection and label‑consistency checks on training data

### 📊 Data & Operations
- Create, list, and retrieve fraud‑scored transactions
- Batch CSV upload – score thousands of transactions at once
- CSV export – filter by date and download results
- Immutable audit log – every prediction written to a separate MongoDB collection; never updated or deleted
- Background tasks for fraud alerts (webhooks) and analytics logging
- Duplicate transaction handling – returns `409 Conflict` instead of crashing

### 📈 MLOps & Monitoring
- **Exploratory Data Analysis (EDA)** – automated data quality report before training (`ml/eda.py`)
- **Feature drift detection** – Population Stability Index (PSI) per feature, with GREEN / YELLOW / RED status
- **Model AUC drift detection** – rolling window comparison across retraining cycles
- **`/admin/drift` endpoint** – admin‑only drift report on the last 500 transactions
- **CI/CD** – GitHub Actions runs lint and tests on every push; free Redis and MongoDB services included

### 🚀 Production Ready
- Deployed on Render (live demo available)
- CORS configured for frontend integration
- Custom middleware for request timing, validation, and session handling
- Structured logging with coloured console output and rotating file logs

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12, FastAPI, Uvicorn |
| Machine Learning | scikit‑learn, XGBoost, SHAP, pandas, numpy |
| Feature Selection | `sklearn.feature_selection.SelectFromModel` |
| Real‑time Features | Redis (sorted sets, time‑windowed) |
| Database | MongoDB (transactions & audit log), SQLite / PostgreSQL (users & API keys) |
| Auth | PyJWT, bcrypt, pyotp, Google OAuth2 |
| Testing | pytest, pytest‑asyncio, httpx |
| CI/CD | GitHub Actions (lint + test on every push) |
| Frontend | React 18, Vite, Tailwind CSS, TanStack Query, Recharts (coming soon) |
| Deployment | Render (API), Vercel / Netlify (frontend) |

---

## 🏗️ System Architecture

```
Client (React dashboard) → https://fraudshield-xjgh.onrender.com
│
├── /api/v1/auth/*          (JWT login, register, Google OAuth)
├── /api/v1/mfa/*           (TOTP setup & verify)
├── /api/v1/api-keys        (manage hashed API keys)
├── /api/v1/predict         (ML + anomaly + SHAP + velocity + imputation)
├── /api/v1/transactions    (list, search)
├── /api/v1/upload          (CSV batch)
├── /api/v1/export          (CSV download)
├── /api/v1/model-info      (admin model diagnostics)
└── /api/v1/admin/drift     (admin feature drift report — PSI per feature)
```

**Predict request flow:**
1. Fetch real‑time velocity features from Redis (time‑windowed sorted sets)
2. Apply statistical imputation if Redis is unavailable (training‑set means)
3. Preprocess with `ColumnTransformer` (scaling + one‑hot encoding)
4. Apply feature selector (keeps only training‑selected features)
5. Score with supervised classifier (XGBoost / RF / etc.)
6. Score with unsupervised anomaly detector (IsolationForest / GMM)
7. Combine scores, apply precision‑recall thresholds → approve / review / block
8. Generate SHAP explanation (top 3 feature contributions)
9. Write to immutable audit log (MongoDB)
10. Record transaction in Redis for future velocity queries

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
```

### 2. Create a virtual environment & activate it
```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env with your real credentials
```

### 5. Start the server
```bash
uvicorn main:app --reload
# API available at http://127.0.0.1:8000
```

### 6. Seed data, run EDA, and train
```bash
# Seed synthetic training data (500 normal + burst attack + mule scenario)
python -m ml.seed_synthetic_data

# Optional: run EDA to inspect data quality before training
python -m ml.eda

# Train all models and save artifacts to ml/models/
python -m ml.train_fraud_model
```

Training produces the following artifacts in `ml/models/`:

| File | Purpose |
|---|---|
| `fraud_model.pkl` | Best supervised classifier |
| `preprocessor.pkl` | Fitted `ColumnTransformer` (scaling + one‑hot) |
| `feature_selector.pkl` | Fitted `SelectFromModel` (keeps important features only) |
| `anomaly_model.pkl` | Best unsupervised anomaly detector |
| `anomaly_bounds.json` | Min/max scores for anomaly normalisation |
| `thresholds.json` | Block / review decision thresholds |
| `feature_names.json` | Selected feature names for SHAP |
| `feature_stats.json` | Training‑set means / medians for serving‑time imputation |
| `feature_reference_dist.json` | Training‑set distributions for PSI drift detection |
| `model_version.txt` | UTC timestamp of training run |
| `auc_history.json` | Rolling AUC history for model drift detection |

---

## 🌍 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `SECRET_KEY` | Yes | – | Secret key for JWT tokens |
| `MONGODB_URL` | Yes | `mongodb://localhost:27017/` | MongoDB connection string |
| `DATABASE_URL` | No | `sqlite:///./test.db` | SQL database (PostgreSQL in prod) |
| `REDIS_URL` | No | `redis://localhost:6379/0` | Redis connection string (velocity features) |
| `GOOGLE_CLIENT_ID` | No | – | Google OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | No | – | Google OAuth client secret |
| `GOOGLE_REDIRECT_URI` | No | `http://localhost:8000/api/v1/auth/google/callback` | OAuth redirect URL |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | `30` | JWT expiration |
| `ALGORITHM` | No | `HS256` | JWT algorithm |
| `ALLOWED_ORIGINS` | No | `http://localhost:5173` | Comma‑separated origins for CORS |

---

## 📚 API Documentation

Once the server is running, visit:
- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

All endpoints are prefixed with `/api/v1`.

**Authentication:**
- `Bearer token` (JWT) for account management and API keys
- `X-API-KEY` header for transaction endpoints

---

## 🧠 Machine Learning Pipeline

The offline training script `ml/train_fraud_model.py` runs the following steps in order:

1. **Load** labelled transactions from MongoDB
2. **Temporal split** – 80% train / 20% val, sorted by time (no future leakage)
3. **Velocity features** – compute historical window aggregations in chronological order
4. **Feature engineering** – `log_amount`, 14‑day per‑category counts
5. **Save feature stats** – training‑set means saved to `feature_stats.json` for serving‑time imputation
6. **Save reference distribution** – raw feature distributions saved to `feature_reference_dist.json` for PSI drift detection
7. **Preprocess** – `ColumnTransformer` (median imputation + standard scaling + one‑hot encoding)
8. **Cluster visualisation** – PCA projection of fraud vs legit to assess feature quality
9. **Training data defenses** – `IsolationForest` outlier removal + label‑flip detection
10. **Feature selection** – `SelectFromModel` (Random Forest importance, median threshold) reduces feature count by ~50%
11. **Semi‑supervised experiment** – K‑Means label propagation from 10% of labelled data
12. **Train 10+ supervised models** – LogReg (Ridge + Lasso), Decision Tree, SVM, Random Forest, ExtraTrees, GBM, XGBoost, Voting, Stacking
13. **Train unsupervised detectors** – IsolationForest and GaussianMixture
14. **AUC drift check** – compare to rolling window of previous training runs
15. **Save all artifacts** – model, preprocessor, selector, thresholds, feature names, version

### Velocity Features (Training ↔ Serving Parity)

All velocity features are computed identically in training and serving:

| Feature | Window | Training source | Serving source |
|---|---|---|---|
| `txn_count_5min` | 5 min | Loop over sorted timestamps | Redis `ZCOUNT` |
| `txn_count_1hr` | 1 hr | Loop over sorted timestamps | Redis `ZCOUNT` |
| `txn_count_24hr` | 24 hr | Loop over sorted timestamps | Redis `ZCOUNT` |
| `amount_sum_24hr` | 24 hr | Windowed sum over amounts | Redis `ZRANGEBYSCORE` → sum |
| `unique_devices_24hr` | 24 hr | `nunique()` over device window | Redis sorted set → `len(set())` |
| `inbound_senders_1hr` | 1 hr | `nunique()` of senders to recipient | Redis `ZCOUNT` on beneficiary key |
| `category_{x}_count_14d` | 14 days | Count of matching `transaction_type` | Redis `ZCOUNT` on per‑category key |

---

## 📈 MLOps & Monitoring

### Exploratory Data Analysis
Run before training to catch data quality issues:
```bash
python -m ml.eda
```
Checks: fraud rate, missing values, amount distribution shape, timestamp span, velocity feature coverage, fraud rate by payment method. Output saved to `ml/reports/eda_summary.csv`.

### Feature Drift Detection (PSI)
Population Stability Index is computed per feature between the training distribution and recent serving traffic:

| PSI Range | Status | Meaning |
|---|---|---|
| < 0.10 | 🟢 GREEN | No significant change |
| 0.10 – 0.25 | 🟡 YELLOW | Moderate shift — worth investigating |
| > 0.25 | 🔴 RED | Significant shift — likely degrading model |

**Check drift:** `GET /api/v1/admin/drift` (admin JWT required). Runs PSI against the last 500 transactions. Report also saved to `ml/reports/drift_report.json`.

### Model AUC Drift Detection
`defenses/drift_detector.py` tracks ROC‑AUC across retraining cycles. If the current run's AUC drops more than 4 points from the rolling mean of the previous 5 runs, a warning is logged. History stored in `ml/models/auc_history.json`.

### Statistical Imputation at Serving Time
If Redis is unavailable, velocity features are filled with training‑set means (from `feature_stats.json`) rather than zeros. This matches the model's learned distribution and avoids biasing scores toward APPROVE for suspicious transactions. Implemented in `ml/feature_stats.py` → `impute_with_stats()`.

---

## 🧪 Running Tests

```bash
# All tests
pytest

# Auth tests only
pytest tests/test_auth.py -v

# Predict tests only
pytest tests/test_predict.py -v
```

The test suite covers:
- User registration and login
- JWT token validation
- API key creation and hashing
- Fraud prediction endpoint (valid key, missing key, invalid key)
- Rule‑based fallback when no ML model is loaded

CI runs these automatically on every push via GitHub Actions (see `.github/workflows/ci.yml`).

---

## 🚢 Deployment

The API is live on Render: https://fraudshield-xjgh.onrender.com

### Deploy your own
1. Fork this repository.
2. Create a new Render Web Service linked to your fork.
3. Set the environment variables (see table above).
4. Render will automatically build and deploy on every push.

A `render.yaml` file is included for Infrastructure‑as‑Code deployment.

### CI/CD
GitHub Actions runs on every push to `main` or `develop` and on every pull request to `main`:
- **Lint** – `ruff check .`
- **Tests** – `pytest tests/` with free Redis and MongoDB service containers

To run the full ML training pipeline manually, uncomment the `train` job in `.github/workflows/ci.yml` and trigger it via the GitHub Actions UI (`workflow_dispatch`).

---

## 📁 Project Structure

```
FraudShield/
├── main.py                        # FastAPI application entry point
├── config.py                      # Pydantic settings
├── requirements.txt
├── render.yaml
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI: lint + test on every push
├── ml/                            # ML pipeline
│   ├── eda.py                     # Exploratory data analysis (run before training)
│   ├── train_fraud_model.py       # End-to-end training pipeline (10+ models)
│   ├── seed_synthetic_data.py     # Generates synthetic training data
│   ├── feature_stats.py           # Save/load training stats for imputation
│   ├── data_drift.py              # PSI-based feature drift detection
│   └── models/                    # Saved model artifacts (git-ignored)
│       ├── fraud_model.pkl
│       ├── preprocessor.pkl
│       ├── feature_selector.pkl
│       ├── anomaly_model.pkl
│       ├── anomaly_bounds.json
│       ├── thresholds.json
│       ├── feature_names.json
│       ├── feature_stats.json
│       ├── feature_reference_dist.json
│       ├── model_version.txt
│       └── auc_history.json
├── ml/reports/                    # Generated reports (git-ignored)
│   ├── eda_summary.csv
│   └── drift_report.json
├── routers/                       # API route handlers
│   ├── auth.py, register.py, google_auth.py
│   ├── api_mgmt.py, predict.py, transaction.py
│   ├── upload.py, download.py, mfa.py
├── crud/                          # Database operations
│   ├── security.py, user_crud.py
│   ├── transaction_crud.py, get_current_user.py
├── schemas/                       # Pydantic models
│   ├── users.py, transaction.py, api_keys.py
├── database/                      # DB connections
│   ├── sql_database.py, mongo.py
├── middleware/                    # Custom ASGI middleware
│   ├── validation.py, timing.py
├── services/                      # Business logic & integrations
│   ├── fraud_service.py
│   ├── redis_client.py            # Time-windowed sorted set velocity features
│   ├── audit_logger.py
│   └── shap_service.py (optional)
├── defenses/                      # Data poisoning detection
│   ├── training_data_scanner.py
│   └── drift_detector.py          # AUC drift across retraining cycles
├── redteam/                       # Label-flip detection
│   └── detect_label_flip.py
├── tests/                         # Integration tests
│   ├── conftest.py, test_auth.py, test_predict.py
├── background_tasks.py            # Async tasks (webhooks, analytics)
├── logger_config.py               # Logging configuration
└── rate_limiter.py                # SlowAPI instance
```

---

## 🗺️ Roadmap

- [x] ML pipeline with multiple models and anomaly detection
- [x] API key hashing and management
- [x] Rate limiting and request validation middleware
- [x] Google OAuth and MFA
- [x] CSV upload and export
- [x] Real‑time velocity features (Redis, time‑windowed sorted sets)
- [x] SHAP explainability on every prediction
- [x] Immutable audit logging
- [x] Data poisoning defenses
- [x] Live deployment on Render
- [x] Exploratory data analysis module (`ml/eda.py`)
- [x] Statistical imputation at serving time (training‑set means)
- [x] Feature selection before training (`SelectFromModel`)
- [x] Feature drift detection (PSI, `/admin/drift` endpoint)
- [x] CI/CD pipeline (GitHub Actions — lint + test)
- [x] 14‑day per‑category velocity features (training + Redis)
- [x] Training‑serving skew fix for `unique_devices_24hr`
- [ ] React frontend dashboard
- [ ] Admin retraining endpoint
- [ ] Full PostgreSQL migration
- [ ] Grafana dashboard for drift and model metrics

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Make your changes and write/update tests.
4. Submit a pull request.

For major changes, open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Built with ❤️ by Dada Duyilemi Israel  
Live demo: https://fraudshield-xjgh.onrender.com  
Frontend coming soon.