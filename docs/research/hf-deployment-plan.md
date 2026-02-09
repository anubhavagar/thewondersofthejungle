# 🦁 Hugging Face Spaces Deployment Plan (Unified Docker)

**Goal**: Deploy the Lion King Health & Gymnastics app to Hugging Face Spaces using a single-container Docker architecture.

## 🛠️ Unified Architecture
Hugging Face Spaces runs a single container. We will build the React frontend and serve it directly from the FastAPI backend.

### 1. Backend Updates (`api/main.py`)
- Prefix all API routes with `/api`.
- Mount the `frontend/dist` directory to serve the React application.
- Listen on port `7860` (HF Default).

### 2. Frontend Updates
- Ensure API calls are made to `/api/...` (already configured in `src/services/api.js`).

### 3. Unified Dockerfile (`Dockerfile.hf`)
- **Stage 1**: Build React Frontend.
- **Stage 2**: Setup Python Environment + Copy Frontend Build + Run FastAPI.

---

## 📄 Proposed `Dockerfile.hf`

```dockerfile
# --- Stage 1: Build React Frontend ---
FROM node:20-slim AS build-frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Final Backend Image ---
FROM python:3.10-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models
RUN python -c "import mediapipe as mp; mp.solutions.pose.Pose(); mp.solutions.face_mesh.FaceMesh()"

# Copy backend code
COPY api ./api
COPY model_service ./model_service

# Copy built frontend from Stage 1
COPY --from=build-frontend /app/frontend/dist ./frontend/dist

# Set environment
ENV PYTHONPATH=/app
ENV PORT=7860

# Run with Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

## 🚀 Deployment Steps
1. Create a new **Space** on Hugging Face.
2. Select **Docker** as the Space SDK.
3. Upload the project files (including the new `Dockerfile` and updated `api/main.py`).
4. HF will automatically build and serve the container.
