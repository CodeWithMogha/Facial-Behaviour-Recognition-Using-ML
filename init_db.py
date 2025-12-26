# init_db.py
import sqlite3

# Connect to SQLite database (creates it if not exists)
conn = sqlite3.connect("user_logs.db")
c = conn.cursor()

# Create logs table
c.execute('''
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        emotion TEXT,
        time TEXT,
        day TEXT,
        date TEXT
    )
''')

conn.commit()
conn.close()
print("Database and table created.")
