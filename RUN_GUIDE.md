# How to Run the Lion King Health Application

Follow this step-by-step guide to get the application running on your computer.

## Prerequisites

Before starting, ensure you have the following installed:

1.  **Python** (3.8 or newer)
    - Check by running: `python --version` in your terminal.
    - If not installed, download from [python.org](https://www.python.org/downloads/).
2.  **Node.js** (Required for the content)
    - Check by running: `node --version` in your terminal.
    - If not installed (or if `npm` command fails), download "LTS" version from [nodejs.org](https://nodejs.org/).

---

## Step 1: Start the Backend (API)

The backend powers the logic for the health analysis.

1.  Open your terminal (Command Prompt or PowerShell).
2.  Navigate to the project folder:
    ```bash
    cd c:\Users\ur_an\projects\let_me_check_your_face_app
    ```
3.  **(Optional but Recommended)** Create a virtual environment:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
4.  Install the required Python libraries:
    ```bash
    pip install -r backend/requirements.txt
    ```
5.  Start the Backend Server:
    ```bash
    python -m uvicorn backend.main:app --reload --port 8000
    ```
    *You should see a message saying "Application startup complete". Keep this terminal window open!*

---

## Step 2: Start the Frontend (User Interface)

The frontend is what you see and interact with in the browser.

1.  Open a **new** terminal window (keep the backend one running).
2.  Navigate to the frontend folder inside the project:
    ```bash
    cd c:\Users\ur_an\projects\let_me_check_your_face_app\frontend
    ```
3.  Install the JavaScript dependencies:
    ```bash
    npm install
    ```
    *(If this command fails, verify you have installed Node.js from the Prerequisites section)*
4.  Start the Frontend Server:
    ```bash
    npm run dev
    ```
5.  Look for a URL in the terminal output, usually:
    `Local: http://localhost:5173/`

---

## Step 3: Use the Application

1.  Open your web browser (Chrome, Edge, Firefox).
2.  Go to the URL from Step 2 (e.g., `http://localhost:5173`).
3.  **Allow Camera Access**: The browser will ask for permission to use your camera. Click "Allow".
4.  **Explore**:
    - **Simba's Eye**: Click "Take Photo" to simulate a face scan.
    - **Circle of Life Data**: Click "Connect Google Fit" to see simulated health data.
    - **Rafiki's Advice**: Read the health insights generated below.

## Troubleshooting

- **Backend won't start?** Make sure you are in the root folder (`let_me_check_your_face_app`), not inside `backend` or `frontend` when running the `uvicorn` command.
- **`npm` not found?** You likely need to install Node.js. Restart your terminal after installing it.
- **Camera not working?** Ensure your browser has permission to access the camera and that no other app is using it.
