import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "The Wonders of the Jungle"
    
    # DB Settings (Hybrid: Switches between local and Vercel Postgres)
    POSTGRES_URL: str = os.getenv("POSTGRES_URL", f"sqlite:///{os.path.join(os.path.dirname(__file__), 'health_app.db')}")
    DB_NAME: str = os.path.join(os.path.dirname(__file__), "health_app.db")
    
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
