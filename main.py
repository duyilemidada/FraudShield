# main.py — fully corrected version
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
import asyncio
from config import settings
from database.mongo import create_indexes
from middleware.timing import RequestTimingMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from rate_limiter import limiter
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

# Base paths for model artifacts
_BASE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(_BASE, 'ml', 'models')

@asynccontextmanager
async def lifespan(app: FastAPI):
    client_logger.info('FraudShield API starting up')
    Base.metadata.create_all(bind=engine)
    await create_indexes()

    model_path           = os.path.join(_MODELS, 'fraud_model.pkl')
    preproc_path         = os.path.join(_MODELS, 'preprocessor.pkl')
    anomaly_path         = os.path.join(_MODELS, 'anomaly_model.pkl')
    anomaly_bounds_path  = os.path.join(_MODELS, 'anomaly_bounds.json')
    thresholds_path      = os.path.join(_MODELS, 'thresholds.json')
    feature_names_path   = os.path.join(_MODELS, 'feature_names.json')
    model_version_path   = os.path.join(_MODELS, 'model_version.txt')

    if os.path.exists(anomaly_path) and os.path.exists(anomaly_bounds_path):
        with open(anomaly_bounds_path, 'r') as f:
            anomaly_bounds = json.load(f)
        app.state.anomaly_model = {
            'model': await asyncio.to_thread(joblib.load, anomaly_path),
            'bounds': anomaly_bounds
        }
        client_logger.info('Anomaly model loaded')
    else:
        app.state.anomaly_model = None

    if os.path.exists(model_path) and os.path.exists(preproc_path):
        try:
            preprocessor = await asyncio.to_thread(joblib.load, preproc_path)
            classifier   = await asyncio.to_thread(joblib.load, model_path)
            client_logger.info(f"Classifier loaded: {type(classifier).__name__}")

            app.state.ml_model = {
                'classifier': classifier,
                'preprocessor': preprocessor,
            }

            if os.path.exists(thresholds_path):
                with open(thresholds_path) as f:
                    app.state.ml_model['thresholds'] = json.load(f)
            else:
                app.state.ml_model['thresholds'] = {
                    "BLOCK_THRESHOLD": 0.75,
                    "REVIEW_THRESHOLD": 0.35
                }

            if os.path.exists(feature_names_path):
                with open(feature_names_path) as f:
                    app.state.ml_model['feature_names'] = json.load(f)
            else:
                app.state.ml_model['feature_names'] = None

            # ── Load feature selector ─────────────────────────────────────────────────
            feature_selector_path = os.path.join(_MODELS, 'feature_selector.pkl')
            if os.path.exists(feature_selector_path) and app.state.ml_model is not None:
                try:
                    app.state.ml_model['feature_selector'] = await asyncio.to_thread(
                        joblib.load, feature_selector_path
                    )
                    client_logger.info("Feature selector loaded.")
                except Exception as e:
                    client_logger.warning(f"Feature selector failed to load: {e}")
                    app.state.ml_model['feature_selector'] = None
            elif app.state.ml_model is not None:
                app.state.ml_model['feature_selector'] = None 

            # ── Load feature statistics for imputation ────────────────────────────────
            # Implements the book's impute_policy={"*": "$mean"}
            from ml.feature_stats import load_feature_stats
            stats_path = os.path.join(_MODELS, 'feature_stats.json')
            app.state.feature_stats = load_feature_stats(_MODELS)           

            try:
                import shap
                model_type = type(classifier).__name__
                if model_type in ['RandomForestClassifier', 'XGBClassifier',
                                  'GradientBoostingClassifier', 'ExtraTreesClassifier']:
                    app.state.ml_model['shap_explainer'] = shap.TreeExplainer(classifier)
                    client_logger.info(f"SHAP loaded for {model_type}")
                else:
                    app.state.ml_model['shap_explainer'] = None
            except Exception as e:
                client_logger.error(f"SHAP failed (non-fatal): {e}")
                app.state.ml_model['shap_explainer'] = None

            app.state.model_version = "unknown"
            if os.path.exists(model_version_path):
                with open(model_version_path) as f:
                    app.state.model_version = f.read().strip()

            client_logger.info("✅ ML model loaded successfully")
        except Exception as e:
            client_logger.error(f"❌ ML model load failed: {e}", exc_info=True)
            app.state.ml_model = None
    else:
        client_logger.warning("⚠️ ML model files not found — rule-based fallback")
        app.state.ml_model = None

    yield
    client_logger.info('FraudShield API shutting down')

# CORS origins
raw_origins = os.getenv("ALLOWED_ORIGINS", "")
ALLOW_ORIGINS = [o.strip() for o in raw_origins.split(",") if o.strip()]
if not ALLOW_ORIGINS:
    ALLOW_ORIGINS = ["http://localhost:5173"]

app = FastAPI(title='FraudShield', lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware order: outermost last → CORS first, then SlowAPI, then Session, then Timing (innermost)
app.add_middleware(RequestTimingMiddleware)                         # innermost
app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-KEY", "X-Requested-With"],
)   # outermost

# TransactionValidationMiddleware is now a dependency in predict.py – removed from global middleware

app.include_router(register_router,   prefix='/api/v1')
app.include_router(auth_router,       prefix='/api/v1')
app.include_router(google_router,     prefix='/api/v1')
app.include_router(api_keys_router,   prefix='/api/v1')
app.include_router(predict_router,    prefix='/api/v1')
app.include_router(download_router,   prefix='/api/v1')
app.include_router(transaction_router,prefix='/api/v1')
app.include_router(mfa_router,        prefix='/api/v1')
app.include_router(upload_router,     prefix='/api/v1')

@app.get('/')
async def root():
    return {'message': '🚀 FraudShield API is running'}