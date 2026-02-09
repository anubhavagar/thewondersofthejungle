# Use official Python 3.10 image
FROM python:3.10

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY api/requirements.txt .

# Install dependencies
# Using --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# # Pre-download ML models (MediaPipe)
# RUN python -c "import mediapipe as mp; mp.solutions.pose.Pose(); mp.solutions.face_mesh.FaceMesh()"

RUN python -c "import mediapipe as mp; \
mp.solutions.pose.Pose(static_image_mode=True); \
mp.solutions.pose.Pose(static_image_mode=False); \
mp.solutions.face_mesh.FaceMesh(static_image_mode=True)"

# Copy the application code
COPY api ./api
COPY model_service ./model_service

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000

# Command to run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
