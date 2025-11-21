import sqlite3

DB_NAME = "prescription.db"

def get_connection():
    conn = sqlite3.connect(DB_NAME)
    return conn

def create_table():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS prescriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_name TEXT,
        date TEXT,
        medicine TEXT,
        dosage TEXT,
        frequency TEXT
    )
    """)
    conn.commit()
    conn.close()
