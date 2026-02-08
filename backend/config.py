import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "The Wonders of the Jungle"
    
    # DB Settings (Local fallback, will be overridden by Vercel Postgres Environment Variables)
    # Vercel Postgres provides POSTGRES_URL, POSTGRES_PRISMA_URL, etc.
    POSTGRES_URL: str = os.getenv("POSTGRES_URL", "sqlite:///./backend/health_app.db")
    
    # OTP Settings
    OTP_EXPIRY_MINUTES: int = 60
    OTP_LENGTH: int = 4
    
    # Simulated Email Settings (For Anubhav)
    OTP_SENDER_EMAIL: str = "anubhavagar@gmail.com"
    
    # Vercel Blob (For image storage)
    BLOB_READ_WRITE_TOKEN: str = os.getenv("BLOB_READ_WRITE_TOKEN", "")

    class Config:
        env_file = ".env"

settings = Settings()
