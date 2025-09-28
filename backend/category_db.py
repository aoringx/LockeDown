import sqlite3
from typing import List, Tuple, Optional

class CategoryDB:
    def __init__(self, db_name: str = "categories.db"):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS items (
        name TEXT NOT NULL,
        category TEXT NOT NULL
        )
        """)
        self.conn.commit()

    def add_category(self, name: str, category: str) -> bool:
        try:
            self.cursor.execute(
                "INSERT INTO items (name, category) VALUES (?, ?)",
                (name, category)
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # URL already exists
    
    def get_category(self, name: str) -> Optional[Tuple[str, str]]:
        """Retrieve a specific URL entry if it exists."""
        self.cursor.execute("SELECT * FROM items WHERE name = ?", (name,))
        return self.cursor.fetchone()

    def close(self):
        """Close the database connection."""
        self.conn.close()