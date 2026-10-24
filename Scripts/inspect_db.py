import sqlite3
import pandas as pd
import os

DB_PATH = "satellite_data.db"

def inspect_db():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file '{DB_PATH}' not found.")
        return

    print(f"Inspecting database: {DB_PATH}")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print(f"Found Tables: {[t[0] for t in tables]}")
    
    for table in tables:
        table_name = table[0]
        # Skip internal sqlite tables if any
        if table_name.startswith('sqlite_'):
            continue
            
        print(f"\n{'='*30} TABLE: {table_name} {'='*30}")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"Total Record Count: {count}")
        
        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [info[1] for info in cursor.fetchall()]
        print(f"Columns: {', '.join(columns)}")
        print("-" * 80)
        
        # Get sample data using pandas for nice formatting
        try:
            df = pd.read_sql_query(f"select count(*) from {table_name}", conn)
            if not df.empty:
                print(df.to_string(index=False))
            else:
                print("(Table is empty)")
        except Exception as e:
            print(f"Error reading table {table_name}: {e}")

    conn.close()

if __name__ == "__main__":
    inspect_db()
