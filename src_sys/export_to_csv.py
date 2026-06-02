# export_to_csv.py
import sys
import os
import asyncio
import pandas as pd

# Add parent directory to path to import database settings
sys.path.append('..')

from sqlalchemy import text
from database import engine

async def export_data():
    # Create directory
    os.makedirs('data/raw', exist_ok=True)
    
    print("Connecting to database...")
    async with engine.connect() as conn:
        # Export interactions
        try:
            print("Exporting interactions...")
            result = await conn.execute(text("SELECT * FROM interactions"))
            rows = result.fetchall()
            if rows:
                columns = result.keys()
                df = pd.DataFrame(rows, columns=columns)
                df.to_csv('data/raw/interactions.csv', index=False)
                print(f"Exported {len(df)} interactions")
            else:
                print("  No data in interactions table")
        except Exception as e:
            print(f"  Error exporting interactions: {e}")
        
        # Export products
        try:
            print("Exporting products...")
            result = await conn.execute(text("SELECT * FROM products"))
            rows = result.fetchall()
            if rows:
                columns = result.keys()
                df = pd.DataFrame(rows, columns=columns)
                df.to_csv('data/raw/produits.csv', index=False)
                print(f"Exported {len(df)} products")
            else:
                print("  No data in products table")
        except Exception as e:
            print(f"  Error exporting products: {e}")
    
    print("\n✅ CSV files created in data/raw/")

if __name__ == "__main__":
    asyncio.run(export_data())