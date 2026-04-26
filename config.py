from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GOOGLE_CLIENT_ID: str
    GOOGLE_CLIENT_SECRET: str
    DATABASE_URL: str = "sqlite:///./test.db"
    MONGODB_URL: str = "mongodb://localhost:27017/"   # ← NEW: easy for prod/Atlas
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/v1/auth/google/callback"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()