import psycopg2
import random
import datetime

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="soil_moisture_db",
    user="postgres",
    password="admin",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# Generate and insert 500 dummy records
for i in range(500):
    image_name = f"image_{i}.jpg"
    random_year = random.randint(2020, 2025)  # Pick a year between 2020 and 2025
    random_month = random.randint(1, 12)  # Random month
    random_day = random.randint(1, 28)  # Keep within 28 to avoid invalid dates
    image_date = datetime.date(random_year, random_month, random_day)
    
    process_date = datetime.datetime.now()  # Current timestamp
    water_percent = round(random.uniform(10, 70), 2)
    dry_soil_percent = round(random.uniform(10, 40), 2)
    wet_soil_percent = round(random.uniform(10, 30), 2)
    salty_muddy_percent = round(random.uniform(1, 25), 2)
    unique_key = int(datetime.datetime.now().timestamp()) + i  # Unique timestamp-based key

    cur.execute("""
        INSERT INTO image_analysis 
        (image_name, image_date, process_date, water_percent, dry_soil_percent, wet_soil_percent, salty_muddy_percent, unique_key) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (image_name, image_date, process_date, water_percent, dry_soil_percent, wet_soil_percent, salty_muddy_percent, unique_key))

# Commit and close
conn.commit()
cur.close()
conn.close()

print("✅ 500 dummy records inserted successfully with a 5-year date range!")
