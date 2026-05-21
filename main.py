# main.py — COMPLETE FIXED VERSION
from fastapi import FastAPI
from contextlib import asynccontextmanager
from database.sql_database import engine, Base
from routers.register import router as register_router
from routers.auth import router as auth_router
from routers.google_auth import router as google_router
from routers.api_mgmt import router as api_keys_router
from routers.predict import router as predict_router
from routers.transaction import router as transaction_router
from routers.mfa import router as mfa_router
from routers.upload import router as upload_router
from routers.download import router as download_router
from logger_config import client_logger
import os
import joblib
import json
from config import settings
from database.mongo import create_indexes
from middleware.timing import RequestTimingMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from rate_limiter import limiter
from fastapi.middleware.cors import CORSMiddleware
from middleware.validation import TransactionValidationMiddleware
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy import text, inspect
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──────────────────────────────────────────
    client_logger.info('FraudShield API starting up')
    Base.metadata.create_all(bind=engine)
    await create_indexes()

    inspector = inspect(engine)
    cols = [c['name'] for c  in inspector.get_columns('api_keys')]
    if 'created_at' not in cols:
        with engine.connect() as conn:
            conn.execute(text('ALTER TABLE api_keys ADD COLUMN created_at DATETIME'))
            conn.commit()

    model_path  = 'ml/models/fraud_model.pkl'
    preproc_path = 'ml/models/preprocessor.pkl'
    anomaly_path  = 'ml/models/anomaly_model.pkl'
    anomaly_bounds_path = 'ml/models/anomaly_bounds.json'
    thresholds_path = 'ml/models/thresholds.json'
   
    if os.path.exists(anomaly_path) and os.path.exists(anomaly_bounds_path):
        with open(anomaly_bounds_path, 'r') as f:
            anomaly_bounds = json.load(f)
        app.state.anomaly_model = {
            'model': joblib.load(anomaly_path),
            'bounds': anomaly_bounds
        }
        client_logger.info('Anomaly model loaded')
    else:
        app.state.anomaly_model = None


    if os.path.exists(model_path) and os.path.exists(preproc_path):  
        try:
            app.state.ml_model = {
                'classifier':   joblib.load(model_path),
                'preprocessor': joblib.load(preproc_path)
            }

            # Load thresholds if available, otherwise fall back to defaults
            if os.path.exists(thresholds_path):
                with open(thresholds_path, 'r') as f:
                    app.state.ml_model['thresholds'] = json.load(f)
                client_logger.info(f"Loaded thresholds: {app.state.ml_model['thresholds']}")
            else :
                 # sensible defaults from earlier guess
                 app.state.ml_model['thresholds'] = {
                    "BLOCK_THRESHOLD": 0.75,
                    "REVIEW_THRESHOLD": 0.35
                 }

            client_logger.info('✅ ML model loaded successfully')
        except Exception as e:
            client_logger.error(f'Failed to load ML model: {e}')
            app.state.ml_model = None
    else:
        client_logger.warning('⚠️  ML model not found — using rule-based fallback')
        app.state.ml_model = None

    yield   

    # ── SHUTDOWN ─────────────────────────────────────────
    client_logger.info('FraudShield API shutting down')

# ── CORS Configuration ─────────────────────────────────
# Use environment variables or sensible defaults
raw_origins = os.getenv("ALLOWED_ORIGINS", "")
ALLOW_ORIGINS = [o.strip() for o in raw_origins.split(",") if o.strip()]

if not ALLOW_ORIGINS:
    ALLOW_ORIGINS = [
        "http://localhost:5173",   
    ]

app = FastAPI(title='FraudShield', lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,                 # allow cookies/Authorization headers
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-KEY", "X-Requested-With"],
)

app.add_middleware(TransactionValidationMiddleware)
app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)

app.add_middleware(RequestTimingMiddleware)

app.include_router(register_router,   prefix='/api/v1')
app.include_router(auth_router,       prefix='/api/v1')
app.include_router(google_router,     prefix='/api/v1')
app.include_router(api_keys_router,   prefix='/api/v1')
app.include_router(predict_router,    prefix='/api/v1')
app.include_router(transaction_router,prefix='/api/v1')
app.include_router(mfa_router,        prefix='/api/v1')
app.include_router(upload_router,     prefix='/api/v1')
app.include_router(download_router,   prefix='/api/v1')

#custom middleware 



@app.get('/')
async def root():
    return {'message': '🚀 FraudShield API is running'}
