import sys
import sqlite3
from pathlib import Path

# Add project root to sys.path to support direct execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Optional
from config import Config

def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Return a configured SQLite connection.
    
    Foreign keys are strictly enforced and rows are returned as sqlite3.Row
    for dict-like access.
    """
    if db_path is None:
        db_path = Config.DATABASE_PATH
        
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize database tables and indexes from schema.sql."""
    Config.ensure_directories()
    schema_path = Config.BASE_DIR / "database" / "schema.sql"
    
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()
        
    conn = get_db_connection(db_path)
    try:
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at", Config.DATABASE_PATH)
