"""
Database helper for storing Copilot session logs and summaries.
"""
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import os

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), 'copilot_logs.db')


def get_connection():
    """Get a database connection with timeout for handling concurrent access."""
    conn = sqlite3.connect(DB_PATH, timeout=60.0, check_same_thread=False)  # 60 second timeout for longer waits
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn


def init_database():
    """Initialize the database schema."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS copilot_sessions (
            session_id TEXT PRIMARY KEY,
            pr_number INTEGER NOT NULL,
            started_at TIMESTAMP,
            ended_at TIMESTAMP,
            exit_code INTEGER,
            status TEXT DEFAULT 'active'
        )
    ''')
    
    # Logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS copilot_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            log_index INTEGER NOT NULL,
            log_line TEXT NOT NULL,
            timestamp TIMESTAMP,
            is_code BOOLEAN DEFAULT 0,
            entry_num INTEGER,
            FOREIGN KEY (session_id) REFERENCES copilot_sessions(session_id)
        )
    ''')
    
    # Summaries table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS copilot_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            summary_text TEXT NOT NULL,
            start_entry_num INTEGER,
            end_entry_num INTEGER,
            timestamp TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES copilot_sessions(session_id)
        )
    ''')
    
    # Create indexes for faster queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_session ON copilot_logs(session_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_summaries_session ON copilot_summaries(session_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_pr ON copilot_sessions(pr_number)')
    
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")


# ============================================================================
# Session Management
# ============================================================================

def create_session(session_id: str, pr_number: int) -> bool:
    """Create a new Copilot session."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO copilot_sessions (session_id, pr_number, started_at, status)
            VALUES (?, ?, ?, 'active')
        ''', (session_id, pr_number, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        logger.info(f"Created session {session_id} for PR #{pr_number}")
        return True
    except sqlite3.IntegrityError:
        # Session already exists
        logger.warning(f"Session {session_id} already exists")
        return False
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return False


def update_session_status(session_id: str, status: str, exit_code: Optional[int] = None):
    """Update session status (active/completed/failed)."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        if status in ['completed', 'failed']:
            cursor.execute('''
                UPDATE copilot_sessions 
                SET status = ?, ended_at = ?, exit_code = ?
                WHERE session_id = ?
            ''', (status, datetime.now().isoformat(), exit_code, session_id))
        else:
            cursor.execute('''
                UPDATE copilot_sessions 
                SET status = ?
                WHERE session_id = ?
            ''', (status, session_id))
        
        conn.commit()
        conn.close()
        logger.info(f"Updated session {session_id} status to {status}")
    except Exception as e:
        logger.error(f"Error updating session status: {e}")


def get_sessions_for_pr(pr_number: int) -> List[Dict]:
    """Get all sessions for a PR."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT session_id, pr_number, started_at, ended_at, exit_code, status
            FROM copilot_sessions
            WHERE pr_number = ?
            ORDER BY started_at DESC
        ''', (pr_number,))
        
        sessions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return sessions
    except Exception as e:
        logger.error(f"Error fetching sessions for PR #{pr_number}: {e}")
        return []


def get_session(session_id: str) -> Optional[Dict]:
    """Get a specific session."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT session_id, pr_number, started_at, ended_at, exit_code, status
            FROM copilot_sessions
            WHERE session_id = ?
        ''', (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        logger.error(f"Error fetching session {session_id}: {e}")
        return None


# ============================================================================
# Log Management
# ============================================================================

def insert_log(session_id: str, log_index: int, log_line: str, 
               is_code: bool = False, entry_num: Optional[int] = None) -> bool:
    """Insert a log line with retry logic for database locks."""
    max_retries = 3
    retry_delay = 0.1  # 100ms
    
    for attempt in range(max_retries):
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO copilot_logs (session_id, log_index, log_line, timestamp, is_code, entry_num)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, log_index, log_line, datetime.now().isoformat(), is_code, entry_num))
            conn.commit()
            conn.close()
            return True
        except sqlite3.OperationalError as e:
            if 'locked' in str(e).lower() and attempt < max_retries - 1:
                logger.warning(f"Database locked on attempt {attempt + 1}, retrying in {retry_delay}s...")
                import time
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            else:
                logger.error(f"Error inserting log after {attempt + 1} attempts: {e}")
                return False
        except Exception as e:
            logger.error(f"Error inserting log: {e}")
            return False
    
    return False


def insert_logs_batch(logs: List[Tuple[str, int, str, bool, Optional[int]]]) -> bool:
    """
    Insert multiple log lines in a single transaction.
    
    Args:
        logs: List of tuples (session_id, log_index, log_line, is_code, entry_num)
    
    Returns:
        True if successful, False otherwise
    """
    if not logs:
        return True
    
    max_retries = 3
    retry_delay = 0.1
    
    for attempt in range(max_retries):
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            # Use executemany for batch insert
            cursor.executemany('''
                INSERT OR IGNORE INTO copilot_logs (session_id, log_index, log_line, timestamp, is_code, entry_num)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', [(s_id, idx, line, datetime.now().isoformat(), is_code, entry_num) 
                  for s_id, idx, line, is_code, entry_num in logs])
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.OperationalError as e:
            if 'locked' in str(e).lower() and attempt < max_retries - 1:
                logger.warning(f"Database locked during batch insert (attempt {attempt + 1}), retrying...")
                import time
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                logger.error(f"Error batch inserting {len(logs)} logs after {attempt + 1} attempts: {e}")
                return False
        except Exception as e:
            logger.error(f"Error batch inserting logs: {e}")
            return False
    
    return False


def get_logs_for_session(session_id: str) -> List[Dict]:
    """Get all logs for a session."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, session_id, log_index, log_line, timestamp, is_code, entry_num
            FROM copilot_logs
            WHERE session_id = ?
            ORDER BY log_index ASC
        ''', (session_id,))
        
        logs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return logs
    except Exception as e:
        logger.error(f"Error fetching logs for session {session_id}: {e}")
        return []


def get_log_count(session_id: str) -> int:
    """Get count of logs for a session without fetching all data."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM copilot_logs WHERE session_id = ?
        ''', (session_id,))
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        logger.error(f"Error counting logs for session {session_id}: {e}")
        return 0


def get_logs_for_entry_range(session_id: str, start_entry: int, end_entry: int) -> List[Dict]:
    """Get logs for a specific entry range."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, session_id, log_index, log_line, timestamp, is_code, entry_num
            FROM copilot_logs
            WHERE session_id = ? AND entry_num >= ? AND entry_num <= ?
            ORDER BY log_index ASC
        ''', (session_id, start_entry, end_entry))
        
        logs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return logs
    except Exception as e:
        logger.error(f"Error fetching logs for entry range: {e}")
        return []


# ============================================================================
# Summary Management
# ============================================================================

def insert_summary(session_id: str, summary_text: str, 
                   start_entry_num: int, end_entry_num: int) -> bool:
    """Insert a summary with retry logic."""
    max_retries = 3
    retry_delay = 0.1
    
    for attempt in range(max_retries):
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO copilot_summaries (session_id, summary_text, start_entry_num, end_entry_num, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, summary_text, start_entry_num, end_entry_num, datetime.now().isoformat()))
            conn.commit()
            conn.close()
            logger.info(f"Inserted summary for session {session_id}, entries {start_entry_num}-{end_entry_num}")
            return True
        except sqlite3.OperationalError as e:
            if 'locked' in str(e).lower() and attempt < max_retries - 1:
                logger.warning(f"Database locked inserting summary (attempt {attempt + 1}), retrying...")
                import time
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                logger.error(f"Error inserting summary after {attempt + 1} attempts: {e}")
                return False
        except Exception as e:
            logger.error(f"Error inserting summary: {e}")
            return False
    
    return False


def get_summaries_for_session(session_id: str) -> List[Dict]:
    """Get all summaries for a session."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, session_id, summary_text, start_entry_num, end_entry_num, timestamp
            FROM copilot_summaries
            WHERE session_id = ?
            ORDER BY start_entry_num ASC
        ''', (session_id,))
        
        summaries = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return summaries
    except Exception as e:
        logger.error(f"Error fetching summaries for session {session_id}: {e}")
        return []


def get_summary_count(session_id: str) -> int:
    """Get count of summaries for a session."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) as count
            FROM copilot_summaries
            WHERE session_id = ?
        ''', (session_id,))
        
        result = cursor.fetchone()
        conn.close()
        return result['count'] if result else 0
    except Exception as e:
        logger.error(f"Error counting summaries: {e}")
        return 0


# Initialize database on module import
init_database()
