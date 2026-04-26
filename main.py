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

@asynccontextmanager
async def lifespan(app: FastAPI):
    client_logger.info("FraudShield API starting up")
    Base.metadata.create_all(bind=engine)  # creates users + api_keys tables
    yield
    client_logger.info("FraudShield API stutting down")

app = FastAPI(title="FraudShield", lifespan=lifespan)

# All routes under /api/v1
app.include_router(register_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")
app.include_router(google_router, prefix="/api/v1")
app.include_router(api_keys_router, prefix="/api/v1")
app.include_router(predict_router, prefix="/api/v1")
app.include_router(transaction_router, prefix="/api/v1")
app.include_router(mfa_router, prefix="/api/v1")
app.include_router(upload_router, prefix="/api/v1")
app.include_router(download_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "🚀 FraudShield API is running - Chapter 4 ready!"}