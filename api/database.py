import sqlite3
import datetime
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .config import settings
import os

# Database Connection Helper
def get_db_connection(retries=10, delay=3):
    uri = settings.POSTGRES_URL
    last_error = None
    
    for i in range(retries):
        try:
            if uri.startswith("postgres://") or uri.startswith("postgresql://"):
                import psycopg2
                from psycopg2.extras import RealDictCursor
                # Vercel provides postgres://, but psycopg2 prefers postgresql://
                if uri.startswith("postgres://"):
                    uri = uri.replace("postgres://", "postgresql://", 1)
                # Use RealDictCursor to match sqlite3.Row behavior
                conn = psycopg2.connect(uri, cursor_factory=RealDictCursor)
                logger.info("Successfully connected to PostgreSQL")
                return conn
            else:
                import sqlite3
                # Fallback to SQLite
                db_path = uri.replace("sqlite:///", "") if uri.startswith("sqlite:///") else uri
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                logger.info(f"Successfully connected to SQLite: {db_path}")
                return conn
        except Exception as e:
            last_error = e
            logger.warning(f"Database connection attempt {i+1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
    
    logger.error(f"Failed to connect to database after {retries} attempts: {last_error}")
    raise last_error

import base64
import uuid

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if we are using Postgres or SQLite
    is_postgres = hasattr(conn, "get_dsn_parameters")
    
    primary_key_type = "SERIAL PRIMARY KEY" if is_postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"
    
    datetime_type = "TIMESTAMP" if is_postgres else "DATETIME"
    json_type = "JSONB" if is_postgres else "TEXT"
    
    # Users Table
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS users (
            id {primary_key_type},
            mobile TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            about TEXT
        )
    ''')
    
    # OTPs Table (Temporary)
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS otps (
            mobile TEXT PRIMARY KEY,
            otp TEXT NOT NULL,
            timestamp {datetime_type} DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # History Table
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS history (
            id {primary_key_type},
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
    
    # --- Gymnastics Specific Tables ---
    
    # Discipline Table (MAG, WAG, etc.)
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS dim_discipline (
            id {primary_key_type},
            name TEXT UNIQUE NOT NULL,
            description TEXT
        )
    ''')
    
    # Apparatus Table (Floor, Vault, Bars, etc.)
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS dim_apparatus (
            id {primary_key_type},
            discipline_id INTEGER,
            name TEXT NOT NULL,
            code TEXT UNIQUE NOT NULL,
            FOREIGN KEY (discipline_id) REFERENCES dim_discipline (id)
        )
    ''')
    
    # Skill Atlas (The Library of Recognized Skills)
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS dim_skill_atlas (
            id {primary_key_type},
            apparatus_id INTEGER,
            skill_name TEXT NOT NULL,
            difficulty_value REAL,
            group_number INTEGER,
            fig_id TEXT UNIQUE,
            technical_requirements {json_type},
            FOREIGN KEY (apparatus_id) REFERENCES dim_apparatus (id)
        )
    ''')
    
    # Biomechanical Thresholds (Angle/Posture data for vision engine)
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS dim_biomechanical_thresholds (
            id {primary_key_type},
            skill_id INTEGER,
            joint_name TEXT NOT NULL,
            ideal_angle REAL,
            min_threshold REAL,
            max_threshold REAL,
            strictness_weight REAL DEFAULT 1.0,
            FOREIGN KEY (skill_id) REFERENCES dim_skill_atlas (id)
        )
    ''')
    
    # Fault Map (What happens when thresholds are missed)
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS dim_fault_map (
            id {primary_key_type},
            fault_code TEXT UNIQUE,
            category TEXT, -- form, technical, landing
            deduction_small REAL,
            deduction_medium REAL,
            deduction_large REAL,
            description TEXT
        )
    ''')
    
    # Assessment Logs (The "Facts" of the analysis)
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS fact_assessment_logs (
            id {primary_key_type},
            user_id INTEGER,
            skill_id INTEGER,
            execution_score REAL,
            difficulty_score REAL,
            total_score REAL,
            deduction_notes {json_type},
            video_url TEXT,
            timestamp {datetime_type} DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (skill_id) REFERENCES dim_skill_atlas (id)
        )
    ''')

    # Migrations (SQLite-only usually, but harmless for first-run Postgres)
    if not is_postgres:
        try:
            cursor.execute("ALTER TABLE history ADD COLUMN image_path TEXT")
        except: pass
        
        try:
            cursor.execute("ALTER TABLE history ADD COLUMN user_id INTEGER")
        except: pass
        
    conn.commit()
    conn.close()

def save_otp(mobile, otp):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Postgres uses ON CONFLICT, SQLite uses INSERT OR REPLACE
    is_postgres = hasattr(conn, "get_dsn_parameters")
    if is_postgres:
        cursor.execute("INSERT INTO otps (mobile, otp, timestamp) VALUES (%s, %s, CURRENT_TIMESTAMP) ON CONFLICT (mobile) DO UPDATE SET otp = EXCLUDED.otp, timestamp = EXCLUDED.timestamp", (mobile, otp))
    else:
        cursor.execute("INSERT OR REPLACE INTO otps (mobile, otp, timestamp) VALUES (?, ?, CURRENT_TIMESTAMP)", (mobile, otp))
    conn.commit()
    conn.close()

def verify_otp_db(mobile, otp, delete_after=True):
    conn = get_db_connection()
    cursor = conn.cursor()
    is_postgres = hasattr(conn, "get_dsn_parameters")
    placeholder = "%s" if is_postgres else "?"
    
    expiry = settings.OTP_EXPIRY_MINUTES
    query = f"SELECT * FROM otps WHERE mobile = {placeholder} AND otp = {placeholder} AND timestamp > datetime('now', '-{expiry} minutes')"
    if is_postgres:
        query = f"SELECT * FROM otps WHERE mobile = %s AND otp = %s AND timestamp > NOW() - INTERVAL '{expiry} minutes'"
        
    cursor.execute(query, (mobile, otp))
    row = cursor.fetchone()
    if row:
        if delete_after:
            cursor.execute(f"DELETE FROM otps WHERE mobile = {placeholder}", (mobile,))
            conn.commit()
        conn.close()
        return True
    conn.close()
    return False

def create_user(mobile, name, about):
    conn = get_db_connection()
    cursor = conn.cursor()
    is_postgres = hasattr(conn, "get_dsn_parameters")
    placeholder = "%s" if is_postgres else "?"
    try:
        cursor.execute(f'''
            INSERT INTO users (mobile, name, about)
            VALUES ({placeholder}, {placeholder}, {placeholder})
        ''', (mobile, name, about))
        conn.commit()
        if is_postgres:
            cursor.execute("SELECT LASTVAL()")
            user_id = cursor.fetchone()[0]
        else:
            user_id = cursor.lastrowid
        return {"id": user_id, "mobile": mobile, "name": name, "about": about}
    except Exception: # Catch IntegrityError etc
        return None
    finally:
        conn.close()

def get_user_by_mobile(mobile):
    conn = get_db_connection()
    is_postgres = hasattr(conn, "get_dsn_parameters")
    placeholder = "%s" if is_postgres else "?"
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM users WHERE mobile = {placeholder}", (mobile,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def save_history(name, result, image_data=None, user_id=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    is_postgres = hasattr(conn, "get_dsn_parameters")
    placeholder = "%s" if is_postgres else "?"
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    saved_image_path = None
    if image_data:
        try:
            if "," in image_data:
                header, encoded = image_data.split(",", 1)
            else:
                encoded = image_data
            file_bytes = base64.b64decode(encoded)
            filename = f"{uuid.uuid4()}.jpg"

            # Check for Vercel Blob Token
            if settings.BLOB_READ_WRITE_TOKEN:
                import vercel_blob
                # Upload to Vercel Blob
                blob = vercel_blob.put(filename, file_bytes, {"type": "image/jpeg"})
                saved_image_path = blob.get("url")
            else:
                # Fallback to local filesystem
                os.makedirs("api/uploads", exist_ok=True)
                filepath = os.path.join("api/uploads", filename)
                with open(filepath, "wb") as f:
                    f.write(file_bytes)
                saved_image_path = f"/uploads/{filename}"
        except Exception as e:
            print(f"Error saving image: {e}")

    cursor.execute(f'''
        INSERT INTO history (user_id, name, timestamp, happiness, stress, hydration, energy, image_path)
        VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
    ''', (user_id, name, timestamp, result.get('happiness'), result.get('stress'), result.get('hydration'), result.get('energy'), saved_image_path))
    conn.commit()
    
    if is_postgres:
        cursor.execute("SELECT LASTVAL()")
        history_id = cursor.fetchone()[0]
    else:
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
    conn = get_db_connection()
    is_postgres = hasattr(conn, "get_dsn_parameters")
    placeholder = "%s" if is_postgres else "?"
    cursor = conn.cursor()
    
    if user_id:
        cursor.execute(f"SELECT * FROM history WHERE user_id = {placeholder} ORDER BY id DESC", (user_id,))
    else:
        cursor.execute("SELECT * FROM history ORDER BY id DESC")
        
    rows = cursor.fetchall()
    conn.close()
    
    history_list = []
    for row in rows:
        # row might be a tuple in Postgres or a Row object in SQLite
        history_list.append({
            "id": row["id"] if not isinstance(row, tuple) else row[0],
            "user_id": row["user_id"] if not isinstance(row, tuple) else row[1],
            "name": row["name"] if not isinstance(row, tuple) else row[2],
            "timestamp": row["timestamp"] if not isinstance(row, tuple) else row[3],
            "image_path": row["image_path"] if not isinstance(row, tuple) else row[8],
            "result": {
                "happiness": row["happiness"] if not isinstance(row, tuple) else row[4],
                "stress": row["stress"] if not isinstance(row, tuple) else row[5],
                "hydration": row["hydration"] if not isinstance(row, tuple) else row[6],
                "energy": row["energy"] if not isinstance(row, tuple) else row[7]
            }
        })
    return history_list
