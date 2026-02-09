from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api.database import init_db
from api.routes.endpoints import router as api_router
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB on start
    try:
        init_db()
    except Exception as e:
        print(f"❌ CRITICAL: Database initialization failed: {e}")
    yield

app = FastAPI(
    title="Lion King Health API", 
    description="Backend for the Lion King themed Health Status App",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router with prefix
app.include_router(api_router, prefix="/api")

# Mount uploads directory
os.makedirs("api/uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="api/uploads"), name="uploads")

# Serve Frontend (SPA)
FRONTEND_PATH = "frontend/dist"
if os.path.exists(FRONTEND_PATH):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_PATH, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # If the request is for an existing file in dist, serve it
        local_path = os.path.join(FRONTEND_PATH, full_path)
        if os.path.isfile(local_path):
            return FileResponse(local_path)
        # Otherwise, return index.html (SPA routing)
        return FileResponse(os.path.join(FRONTEND_PATH, "index.html"))
else:
    @app.get("/")
    async def root():
        return {"message": "Welcome to the Pride Lands API (Frontend build not found)"}

