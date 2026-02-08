import sqlite3
import datetime

import os
DB_NAME = os.path.join(os.path.dirname(__file__), "health_app.db")
import base64
import uuid

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Users Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mobile TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            about TEXT
        )
    ''')
    
    # OTPs Table (Temporary)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS otps (
            mobile TEXT PRIMARY KEY,
            otp TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # History Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            happiness TEXT,
            stress TEXT,
            hydration TEXT,
            energy TEXT,
            image_path TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Migrations
    try:
        cursor.execute("ALTER TABLE history ADD COLUMN image_path TEXT")
    except sqlite3.OperationalError: pass
    
    try:
        cursor.execute("ALTER TABLE history ADD COLUMN user_id INTEGER")
    except sqlite3.OperationalError: pass
        
    conn.commit()
    conn.close()

def save_otp(mobile, otp):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO otps (mobile, otp, timestamp) VALUES (?, ?, CURRENT_TIMESTAMP)", (mobile, otp))
    conn.commit()
    conn.close()

def verify_otp_db(mobile, otp, delete_after=True):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # OTP valid for 60 minutes
    cursor.execute("SELECT * FROM otps WHERE mobile = ? AND otp = ? AND timestamp > datetime('now', '-60 minutes')", (mobile, otp))
    row = cursor.fetchone()
    if row:
        if delete_after:
            cursor.execute("DELETE FROM otps WHERE mobile = ?", (mobile,))
            conn.commit()
        conn.close()
        return True
    conn.close()
    return False

def create_user(mobile, name, about):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO users (mobile, name, about)
            VALUES (?, ?, ?)
        ''', (mobile, name, about))
        conn.commit()
        user_id = cursor.lastrowid
        return {"id": user_id, "mobile": mobile, "name": name, "about": about}
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def get_user_by_mobile(mobile):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE mobile = ?", (mobile,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def save_history(name, result, image_data=None, user_id=None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    saved_image_path = None
    if image_data:
        try:
            os.makedirs("backend/uploads", exist_ok=True)
            if "," in image_data:
                header, encoded = image_data.split(",", 1)
            else:
                encoded = image_data
            file_bytes = base64.b64decode(encoded)
            filename = f"{uuid.uuid4()}.jpg"
            filepath = os.path.join("backend/uploads", filename)
            with open(filepath, "wb") as f:
                f.write(file_bytes)
            saved_image_path = f"/uploads/{filename}"
        except Exception as e:
            print(f"Error saving image: {e}")

    cursor.execute('''
        INSERT INTO history (user_id, name, timestamp, happiness, stress, hydration, energy, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, name, timestamp, result.get('happiness'), result.get('stress'), result.get('hydration'), result.get('energy'), saved_image_path))
    conn.commit()
    history_id = cursor.lastrowid
    conn.close()
    return {
        "id": history_id,
        "user_id": user_id,
        "name": name,
        "timestamp": timestamp,
        "result": result,
        "image_path": saved_image_path
    }

def get_history(user_id=None):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if user_id:
        cursor.execute("SELECT * FROM history WHERE user_id = ? ORDER BY id DESC", (user_id,))
    else:
        cursor.execute("SELECT * FROM history ORDER BY id DESC")
        
    rows = cursor.fetchall()
    conn.close()
    
    history_list = []
    for row in rows:
        history_list.append({
            "id": row["id"],
            "user_id": row["user_id"],
            "name": row["name"],
            "timestamp": row["timestamp"],
            "image_path": row["image_path"],
            "result": {
                "happiness": row["happiness"],
                "stress": row["stress"],
                "hydration": row["hydration"],
                "energy": row["energy"]
            }
        })
    return history_list
