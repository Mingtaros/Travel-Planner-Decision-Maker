"""
Script to load CSV data into MySQL database using Python
This approach is more reliable than direct LOAD DATA INFILE
"""

import os
import csv
import mysql.connector
from mysql.connector import Error

# Database connection parameters
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'planner',
    'password': 'plannerpassword',
    'database': 'ai_planning_project'
}

# Path to CSV files
CSV_DIR = './data'

def load_attractions():
    """Load data into attractions table"""
    # file_path = os.path.join(CSV_DIR, 'attractions.csv')
    file_path = CSV_DIR + '/locationData/attractions.csv'
    query = """
    INSERT INTO attractions (aid, aname, expenditure, timespent)
    VALUES (%s, %s, %s, %s)
    """
    
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        records = [(int(row[0]), row[1], int(row[2]) if row[2] else None, 
                   float(row[3]) if row[3] else None) for row in csv_reader]
    
    return query, records

def load_tourists():
    """Load data into tourists table"""
    # file_path = os.path.join(CSV_DIR, 'tourists.csv')
    file_path = CSV_DIR + '/locationData/tourists.csv'
    query = """
    INSERT INTO tourists (tid, type)
    VALUES (%s, %s)
    """
    
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        records = [(int(row[0]), row[1]) for row in csv_reader]
    
    return query, records

def load_foodcentre():
    """Load data into foodcentre table"""
    # file_path = os.path.join(CSV_DIR, 'foodcentre.csv')
    file_path = CSV_DIR + '/locationData/foodcentre.csv'
    query = """
    INSERT INTO foodcentre (fid, name, expenditure, timespent, rating, type, bestfor, highlights, address)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        records = []
        for row in csv_reader:
            # Extend row if necessary to ensure it has enough elements
            extended_row = row + [None] * (9 - len(row))
            # Convert values to appropriate types
            record = (
                int(extended_row[0]) if extended_row[0] else None,
                extended_row[1],
                int(extended_row[2]) if extended_row[2] else None,
                float(extended_row[3]) if extended_row[3] else None,
                float(extended_row[4]) if extended_row[4] else None,
                extended_row[5],
                extended_row[6],
                extended_row[7],
                extended_row[8]
            )
            records.append(record)
    
    return query, records

def load_preference():
    """Load data into preference table"""
    # file_path = os.path.join(CSV_DIR, 'preference.csv')
    file_path = CSV_DIR + '/locationData/preference.csv'
    query = """
    INSERT INTO preference (type, aname, scale)
    VALUES (%s, %s, %s)
    """
    
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        records = [(row[0], row[1], int(row[2]) if row[2] else None) for row in csv_reader]
    
    return query, records

def load_transit_fare():
    """Load data into fare table"""
    file_path = CSV_DIR + '/transitData/transit_fare_data.csv'
    query = """
    INSERT INTO transit_fare (id, lower_distance,upper_distance,basic_fare,express_fare)
    VALUES (%s, %s, %s, %s, %s)
    """
    
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        records = [(float(row[0]) if row[0] else None, 
                    float(row[1]) if row[1] else None,
                    float(row[2]) if row[2] else None,
                    float(row[3]) if row[3] else None,
                    float(row[4]) if row[4] else None) for row in csv_reader]
    
    return query, records

def insert_data(connection, query, records):
    """Insert records using the provided query"""
    try:
        cursor = connection.cursor()
        cursor.executemany(query, records)
        connection.commit()
        print(f"Inserted {cursor.rowcount} rows")
        return True
    except Error as e:
        print(f"Error: {e}")
        connection.rollback()
        return False
    finally:
        cursor.close()

def main():
    """Main function to load all data"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        print("Connected to MySQL database")
        
        # Load data for each table
        for loader_func in [load_attractions, load_tourists, load_foodcentre, load_preference, load_transit_fare]:
            table_name = loader_func.__name__.replace('load_', '')
            print(f"\nLoading {table_name}...")
            query, records = loader_func()
            if records:
                success = insert_data(connection, query, records)
                if success:
                    print(f"Successfully loaded {len(records)} records into {table_name}")
            else:
                print(f"No data found for {table_name}")
                
        # Verify data was loaded
        cursor = connection.cursor()
        for table in ['attractions', 'tourists', 'foodcentre', 'preference']:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"{table.capitalize()} count: {count}")
        cursor.close()
            
    except Error as e:
        print(f"Database connection error: {e}")
    finally:
        if connection and connection.is_connected():
            connection.close()
            print("\nDatabase connection closed")

if __name__ == "__main__":
    main()