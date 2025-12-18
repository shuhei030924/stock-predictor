import sqlite3
import pandas as pd

def check_db(db_name):
    db_path = f'data/{db_name}'
    print(f"\n\n========== Checking {db_name} ==========")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("--- Tables ---")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        for t in tables:
            print(t[0])
            
        # Check for price tables (usually contain 'price' or 'rate')
        price_tables = [t[0] for t in tables if 'price' in t[0] or 'rate' in t[0] or 'cache' in t[0]]
        
        # Explicitly check for price_cache_1h if not found automatically
        if 'price_cache_1h' not in price_tables and 'price_cache_1h' in [t[0] for t in tables]:
             price_tables.append('price_cache_1h')

        for table in price_tables:
            print(f"\n--- Checking {table} for time components ---")
            try:
                # Check if any date string is longer than 10 chars
                # We assume the column name is 'date' or 'timestamp' or 'time'
                # First get columns
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [info[1] for info in cursor.fetchall()]
                date_col = next((c for c in columns if 'date' in c.lower() or 'time' in c.lower()), None)
                
                if date_col:
                    print(f"Date column identified as: {date_col}")
                    cursor.execute(f"SELECT {date_col} FROM {table} WHERE length({date_col}) > 10 LIMIT 5")
                    rows = cursor.fetchall()
                    if rows:
                        print(f"Found rows with time component in {table}:")
                        for r in rows:
                            print(r[0])
                    else:
                        print(f"No rows with time component found in '{table}'.")
                        
                    # Check sample data
                    print(f"\n--- Sample Data from {table} ---")
                    cursor.execute(f"SELECT * FROM {table} ORDER BY {date_col} DESC LIMIT 5")
                    cols = [description[0] for description in cursor.description]
                    rows = cursor.fetchall()
                    for r in rows:
                        print(dict(zip(cols, r)))
                else:
                    print(f"No date/time column found in {table}")

            except Exception as e:
                print(f"Error checking {table}: {e}")

        conn.close()
    except Exception as e:
        print(f"Could not connect to {db_name}: {e}")

if __name__ == "__main__":
    check_db('stock_predictor.db')
    check_db('stock_data.db')
