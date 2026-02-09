# --- Stage 1: Build React Frontend ---
FROM node:20-slim AS build-frontend
WORKDIR /app/frontend

# Copy frontend dependency files
COPY frontend/package*.json ./
RUN npm install

# Copy frontend source and build
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Final Backend Image ---
FROM python:3.10-slim
WORKDIR /app

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ML models for faster Space startup
RUN python -c "import mediapipe as mp; \
mp.solutions.pose.Pose(static_image_mode=True); \
mp.solutions.face_mesh.FaceMesh(static_image_mode=True)"

# Copy backend and model service code
COPY api ./api
COPY model_service ./model_service

# Copy the built frontend from Stage 1 into the location FastAPI expects
COPY --from=build-frontend /app/frontend/dist ./frontend/dist

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=7860

# Expose the HF default port
EXPOSE 7860

# Run the unified application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
