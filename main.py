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
from database.mongo import create_indexes
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──────────────────────────────────────────
    client_logger.info('FraudShield API starting up')
    Base.metadata.create_all(bind=engine)
    await create_indexes()

    model_path  = 'ml/models/fraud_model.pkl'
    preroc_path = 'ml/models/preprocessor.pkl'
    thresholds_path = 'ml/models/thresholds.json'
   
    if os.path.exists(model_path) and os.path.exists(preroc_path):  
        try:
            app.state.ml_model = {
                'classifier':   joblib.load(model_path),
                'preprocessor': joblib.load(preroc_path)
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

app = FastAPI(title='FraudShield', lifespan=lifespan)

app.include_router(register_router,   prefix='/api/v1')
app.include_router(auth_router,       prefix='/api/v1')
app.include_router(google_router,     prefix='/api/v1')
app.include_router(api_keys_router,   prefix='/api/v1')
app.include_router(predict_router,    prefix='/api/v1')
app.include_router(transaction_router,prefix='/api/v1')
app.include_router(mfa_router,        prefix='/api/v1')
app.include_router(upload_router,     prefix='/api/v1')
app.include_router(download_router,   prefix='/api/v1')

@app.get('/')
async def root():
    return {'message': '🚀 FraudShield API is running'}
