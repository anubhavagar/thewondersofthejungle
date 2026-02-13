from pydantic import BaseModel
from typing import Optional, Dict

class FaceAnalysisRequest(BaseModel):
    image: str  # Base64 encoded image

class HealthDataRequest(BaseModel):
    steps: int
    heart_rate: int

class AnalysisResponse(BaseModel):
    # Flexible dict to accommodate various metrics like "happiness", "advice", etc.
    # Flexible dict to accommodate various metrics like "happiness", "advice", etc.
    results: Dict[str, str]

class HistoryRequest(BaseModel):
    name: str
    result: Dict[str, str]
    image: Optional[str] = None # Base64 encoded image
    user_id: Optional[int] = None

class GymnasticsAnalysisRequest(BaseModel):
    media_data: str # Base64 encoded image or video (or dummy identifier for sim)
    media_type: str # 'image' or 'video'
    category: Optional[str] = "Senior" # e.g. "Under 10", "Junior", "Senior"
    hold_duration: Optional[float] = 2.0 # Minimum 2s for full credit

class OTPRequest(BaseModel):
    mobile: str

class OTPVerifyRequest(BaseModel):
    mobile: str
    otp: str
    name: Optional[str] = None
    about: Optional[str] = None

class AuthResponse(BaseModel):
    user_id: int
    mobile: str
    name: str
    about: str
    token: str # Simulated token (just user_id for now)
    is_new_user: bool = False
