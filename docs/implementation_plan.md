# Health Status Application Implementation Plan

## Goal Description
Create a responsive web application that allows users (specifically targeting kids with a "Lion King" theme) to capture an image (Face Scan) and connect health data (Google Fit). The backend analyzes this data to provide health advice.

## User Review Required
> [!IMPORTANT]
> **Google Fit Integration**: Since we cannot easily authenticate with real Google Fit APIs in this local environment without a registered GCP project, we will **simulate** the connection and data retrieval (e.g., clicking "Connect" will auto-fill dummy data).
> **Face Scan**: The "State of the Art" face scan will be a simulated analysis for this prototype, processing the image to return mock vital signs.

## Proposed Changes

### Project Structure
- `backend/`: FastAPI application.
- `frontend/`: React + Vite + Bootstrap application.
- `model_service/`: Logic for Face Scan and Data analysis.

### 1. Backend (FastAPI)
- **`backend/main.py`**: Entry point.
- **`backend/api/endpoints.py`**:
    - `POST /analyze/face`: Accepts image, returns "face scan" results.
    - `POST /analyze/wellness`: Accepts Google Fit data JSON, returns advice.
- **`backend/schemas/`**: Pydantic models (HealthData, AnalysisResult).

### 2. Frontend (React + Vite + Bootstrap)
- **Design System**: "Lion King" Theme.
    - Colors: Sunset Orange, Gold, Earthy Browns.
    - Typography: Playful implementation using Bootstrap classes and custom CSS.
- **Components**:
    - `App.jsx`: Main layout "Pride Lands" container.
    - `CameraCapture.jsx`: "Simba's Eye" - Camera interface.
    - `GoogleFitConnect.jsx`: "Circle of Life Data" - Simulated connector.
    - `HealthTable.jsx`: "Rafiki's Advice" - Results display with feedback buttons.
- **Dependencies**: `bootstrap`, `react-bootstrap` (or standard bootstrap JS), `axios`.

### 3. Model Service (Python)
- **`model_service/vision.py`**: Mock logic to detect "vital signs" from image.
- **`model_service/health.py`**: Logic to analyze step count/heart rate (Google Fit data).

## Verification Plan

### Automated Tests
- Backend `pytest` for data processing logic.

### Manual Verification
1. **Theme Check**: Verify the app looks like the Lion King theme (Orange/Gold).
2. **Responsiveness**: Resize window to ensure Bootstrap handles mobile/desktop views.
3. **Google Fit**: Click "Connect Google Fit" -> Verify dummy data appears.
4. **Face Scan**: Take a photo -> Verify "Analysis" returns stats (e.g., "Happiness Level: High", "Hydration: Good").
5. **Feedback**: Click "Helpful?" button in results -> Verify UI update.
