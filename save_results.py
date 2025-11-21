from db import get_connection, create_table

def save_row(patient, date, medicine, dosage, frequency):
    create_table()
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO prescriptions (patient_name, date, medicine, dosage, frequency)
        VALUES (?, ?, ?, ?, ?)
    """, (patient, date, medicine, dosage, frequency))

    conn.commit()
    conn.close()
