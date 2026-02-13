from fastapi import APIRouter, HTTPException
from typing import Optional
from api.schemas.schemas import FaceAnalysisRequest, HealthDataRequest, HistoryRequest, GymnasticsAnalysisRequest, OTPRequest, OTPVerifyRequest, AuthResponse
from model_service.vision import vision_model
from model_service.health import health_model
from model_service.gymnastics import gymnastics_analyzer # Real implementation
from api.database import init_db, save_history, get_history, create_user, get_user_by_mobile, save_otp, verify_otp_db, list_tables
from api.config import settings

router = APIRouter()

@router.get("/diagnostics/db")
async def db_diagnostics():
    """Returns a list of tables and their row counts."""
    try:
        tables = list_tables()
        return {"tables": tables, "count": len(tables)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/auth/request-otp")
async def request_otp(request: OTPRequest):
    try:
        # Generate OTP based on config length
        import random
        otp = "".join([str(random.randint(0, 9)) for _ in range(settings.OTP_LENGTH)])
        
        # Save to DB
        from api.database import save_otp
        save_otp(request.mobile, otp)
        
        # Simulation: Print to console (SMS)
        print("\n" + "="*30)
        print(f"🦁 JUNGLE OTP FOR {request.mobile}: {otp}")
        
        # Simulation: Send to Email (as requested)
        print(f"📧 EMAIL SIMULATION: Sent to {settings.OTP_SENDER_EMAIL}")
        print("="*30 + "\n")
        
        return {"message": f"OTP sent to {request.mobile} and {settings.OTP_SENDER_EMAIL} (Simulated)"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate OTP: {str(e)}")

@router.post("/auth/verify-otp")
async def verify_otp(request: OTPVerifyRequest):
    from api.database import verify_otp_db, get_user_by_mobile, create_user
    
    user = get_user_by_mobile(request.mobile)
    
    # If it's a new user and they haven't provided name/about yet, don't delete yet
    # Handle both None and empty strings from frontend
    is_step_two = (user is None and not request.name)
    should_delete = not is_step_two
    
    if not verify_otp_db(request.mobile, request.otp, delete_after=should_delete):
        raise HTTPException(status_code=401, detail="Invalid or expired OTP")
    
    is_new_user = False
    
    if not user:
        if not request.name:
            # Need registration info
            return {"is_new_user": True, "mobile": request.mobile}
        
        user = create_user(request.mobile, request.name, request.about)
        is_new_user = True
    
    return AuthResponse(
        user_id=user["id"],
        mobile=user["mobile"],
        name=user["name"],
        about=user["about"],
        token=str(user["id"]),
        is_new_user=is_new_user
    )

@router.get("/history")
async def read_history(user_id: Optional[int] = None):
    return get_history(user_id)

@router.post("/history")
async def create_history(request: HistoryRequest):
    return save_history(request.name, request.result, request.image, request.user_id)

@router.post("/analyze/face")
async def analyze_face(request: FaceAnalysisRequest):
    try:
        # In a real app, we would decode the base64 image here
        # For simulation, we just pass it (or ignore it) and return mock data
        results = vision_model.analyze_face(request.image)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/gymnastics")
async def analyze_gymnastics(request: GymnasticsAnalysisRequest):
    try:
        results = gymnastics_analyzer.analyze_media(
            media_data=request.media_data, 
            media_type=request.media_type, 
            category=request.category,
            hold_duration=request.hold_duration
        )
        return results
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(error_msg)
        with open("backend_error.log", "w") as f:
            f.write(error_msg)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/wellness")
async def analyze_wellness(request: HealthDataRequest):
    try:
        data = {"steps": request.steps, "heart_rate": request.heart_rate}
        results = health_model.analyze_data(data)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
