import sqlite3
from pathlib import Path

DB_PATH = Path("database/incidents.db")


def get_connection():
    DB_PATH.parent.mkdir(exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_tables():

    conn = get_connection()

    cur = conn.cursor()

    cur.execute("""

    CREATE TABLE IF NOT EXISTS incidents(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        title TEXT NOT NULL,

        category TEXT,

        affected_service TEXT,

        assigned_team TEXT,

        severity TEXT,

        symptoms TEXT,

        root_cause TEXT,

        resolution TEXT,

        status TEXT,

        created_at TEXT

    )

    """)

    cur.execute("""

    CREATE TABLE IF NOT EXISTS feedback(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        question TEXT,

        answer TEXT,

        feedback TEXT,

        created_at TEXT

    )

    """)

    conn.commit()
    conn.close()


def insert_incident(
        title,
        category,
        affected_service,
        assigned_team,
        severity,
        symptoms,
        root_cause,
        resolution,
        status,
        created_at):

    conn = get_connection()

    cur = conn.cursor()

    cur.execute("""

    INSERT INTO incidents(

        title,
        category,
        affected_service,
        assigned_team,
        severity,
        symptoms,
        root_cause,
        resolution,
        status,
        created_at

    )

    VALUES(?,?,?,?,?,?,?,?,?,?)

    """,

    (

        title,
        category,
        affected_service,
        assigned_team,
        severity,
        symptoms,
        root_cause,
        resolution,
        status,
        created_at

    ))

    conn.commit()

    conn.close()

def get_all_incidents():

    conn = get_connection()

    cur = conn.cursor()

    cur.execute("SELECT * FROM incidents ORDER BY id DESC")

    rows = cur.fetchall()

    conn.close()

    return rows


def search_incidents(keyword):

    conn = get_connection()

    cur = conn.cursor()

    cur.execute("""

    SELECT *

    FROM incidents

    WHERE

    title LIKE ?

    OR category LIKE ?

    OR symptoms LIKE ?

    OR root_cause LIKE ?

    OR resolution LIKE ?

    """,

    (
        f"%{keyword}%",
        f"%{keyword}%",
        f"%{keyword}%",
        f"%{keyword}%",
        f"%{keyword}%"
    ))

    rows = cur.fetchall()

    conn.close()

    return rows


def insert_feedback(question, answer, feedback, created_at):

    conn = get_connection()

    cur = conn.cursor()

    cur.execute("""

    INSERT INTO feedback(

        question,
        answer,
        feedback,
        created_at

    )

    VALUES(?,?,?,?)

    """,

    (
        question,
        answer,
        feedback,
        created_at
    ))

    conn.commit()

    conn.close()


def get_feedback_stats():

    conn = get_connection()

    cur = conn.cursor()

    cur.execute("""

    SELECT feedback, COUNT(*) as count
    FROM feedback
    GROUP BY feedback

    """)

    rows = cur.fetchall()

    conn.close()

    return rows