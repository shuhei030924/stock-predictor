import sqlite3
import pandas as pd
from database.db_manager import DatabaseManager

def check_data():
    db = DatabaseManager()
    conn = db._get_connection()
    cursor = conn.cursor()
    
    print("Checking 'price_cache' table...")
    try:
        cursor.execute("SELECT count(*) FROM price_cache")
        count = cursor.fetchone()[0]
        print(f"Total rows: {count}")
        
        cursor.execute("SELECT * FROM price_cache LIMIT 5")
        rows = cursor.fetchall()
        print("Sample rows:")
        for row in rows:
            print(dict(row))
            
        # Check for time components in date to infer 1h data
        cursor.execute("SELECT date FROM price_cache WHERE length(date) > 10 LIMIT 5")
        time_rows = cursor.fetchall()
        if time_rows:
            print(f"\nFound {len(time_rows)} rows with time component (likely 1h data):")
            for row in time_rows:
                print(row[0])
        else:
            print("\nNo rows with time component found (likely only Daily data).")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_data()
