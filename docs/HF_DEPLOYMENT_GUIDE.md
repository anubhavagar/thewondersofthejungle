# 🚀 Deploying to Hugging Face Spaces (Docker)

I have prepared your project for a unified, single-container deployment suitable for Hugging Face Spaces.

## 🧱 What I've Changed
1.  **Created a Root `Dockerfile`**: A multi-stage build that compiles the React frontend and packages it with the FastAPI backend.
2.  **Updated `api/main.py`**:
    -   Configured the API to use the `/api` prefix (matching the frontend expectations).
    -   Added logic to serve the frontend's static files and handle SPA (Single Page Application) routing.
3.  **Hugging Face Ready**: The app now listens on port `7860` (the Hugging Face default).

---

## 🛠️ Step-by-Step Deployment Guide

### 1. Create a New Hugging Face Space
- Go to [huggingface.co/new-space](https://huggingface.co/new-space).
- **Space Name**: e.g., `lion-king-health`.
- **License**: Choose your preference (e.g., Apache 2.0).
- **SDK**: Select **Docker**.
- **Template**: Select **Blank**.

### 2. Push Your Code
Hugging Face Spaces act as a Git repository. You can push your code using Git:

```bash
# 1. Initialize git if you haven't
git init
git add .
git commit -m "Deployment ready: Unified Docker architecture"

# 2. Add HF as a remote (Replace YOUR_USERNAME and SPACE_NAME)
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME

# 3. Push to Hugging Face
git push --force hf main
```

### 3. Configure External Database (Recommended)
By default, the app will use **SQLite** inside the container. This is great for a demo, but your data will be cleared every time the Space restarts.

**For persistence**:
1.  Set up a free Postgres DB (e.g., on [Neon.tech](https://neon.tech/) or [Supabase](https://supabase.com/)).
2.  In your HF Space, go to **Settings** -> **Variables and secrets**.
3.  Add a **Secret** named `POSTGRES_URL` with your connection string.

---

## 🎯 Verification
Once the build is complete (you can monitor it in the HF "Logs" tab), your app will be live at:
`https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME`

### 💡 Pro Tip:
If you want to keep the app "awake", you can enable the "Sleep Time" skip in HF Settings (requires an HF account tier). Otherwise, the app will pause after 48 hours of inactivity.

Enjoy your global "Digital Judge"! 🦁🌍🐘
