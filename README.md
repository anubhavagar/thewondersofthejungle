# Lion King Health Tracker 🦁

A responsive web application that allows users to track their health with a fun Lion King theme!

## Project Structure

- `backend/`: Python FastAPI application for analysis logic.
- `frontend/`: React + Vite application for the user interface.
- `model_service/`: Simulated AI model logic for health analysis.

## Prerequisites

- **Python 3.8+**
- **Node.js** and **npm** (Required for Frontend)

## Setup Instructions

### 1. Backend Setup

Navigate to the project root and install dependencies:

```bash
pip install -r backend/requirements.txt
```

Run the backend server:

```bash
python -m uvicorn backend.main:app --reload --port 8000
```
The API will be available at `http://localhost:8000`.

### 2. Frontend Setup

Navigate to the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```
*(Note: If `npm` is not installed, please install Node.js from [nodejs.org](https://nodejs.org/))*

Run the development server:

```bash
npm run dev
```
Open your browser to the URL shown (usually `http://localhost:5173`).

## Features

- **Lion King Theme**: Immerse yourself in the Pride Lands with custom colors and fonts.
- **Simba's Eye (Camera)**: Capture a photo for a simulated health scan.
- **Circle of Life Data**: Connect (simulate) Google Fit data.
- **Rafiki's Advice**: Get personalized health feedback based on your scan and data.
