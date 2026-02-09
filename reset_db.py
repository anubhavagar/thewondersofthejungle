import sys
import os

# Add the project root to sys.path to import api modules
sys.path.append(os.getcwd())

try:
    from api.database import get_db_connection
    from api.config import settings
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def reset_database():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if we are using Postgres or SQLite
        is_postgres = hasattr(conn, "get_dsn_parameters")
        
        tables = ["history", "otps", "users"]
        
        logger.info(f"Connected to {'PostgreSQL' if is_postgres else 'SQLite'}")
        
        for table in tables:
            try:
                if is_postgres:
                    # TRUNCATE is faster and resets identities in Postgres
                    cursor.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;")
                else:
                    cursor.execute(f"DELETE FROM {table};")
                    cursor.execute(f"DELETE FROM sqlite_sequence WHERE name='{table}';")
                logger.info(f"Successfully emptied table: {table}")
            except Exception as e:
                 logger.error(f"Error emptying table {table}: {e}")
        
        conn.commit()
        conn.close()
        logger.info("Database reset complete. All records cleared.")

    if __name__ == "__main__":
        # Manual override for local run if needed
        # os.environ["POSTGRES_URL"] = "postgresql://user:password@localhost:5432/health_db"
        reset_database()

except ImportError as e:
    print(f"Error: Could not import api modules. Run this from the project root. {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
