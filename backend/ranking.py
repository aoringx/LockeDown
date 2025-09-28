import sqlite3
from typing import List, Tuple, Optional

class Ranking:
    def __init__(self, db_name: str = "ranking.db"):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS players (
        name TEXT NOT NULL,
        score REAL NOT NULL
        )
        """)
        self.conn.commit()

    def add_player(self, name: str, score: float) -> bool:
        if self.get_score(name) is not None:
            return False  # Player already exists
        try:
            self.cursor.execute(
                "INSERT OR IGNORE INTO players (name, score) VALUES (?, ?)",
                (name, score)
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False 
    
    def get_ranking(self, name: str) -> Optional[Tuple[str, str]]:
        self.cursor.execute("SELECT name, score FROM players ORDER BY score DESC")
        ranked_players = self.cursor.fetchall()
    
    def get_score(self, name: str) -> Optional[float]:
        self.cursor.execute("SELECT score FROM players WHERE name = ?", (name,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def set_score(self, name: str, new_score: float) -> bool:
        self.cursor.execute(
            "UPDATE players SET score = ? WHERE name = ?",
            (new_score, name)
        )
        self.conn.commit()
        return self.cursor.rowcount > 0
    
    def increment_score(self, name: str, increment: float) -> bool:
        self.cursor.execute(
            "UPDATE players SET score = score + ? WHERE name = ?",
            (increment, name)
        )
        self.conn.commit()
        return self.cursor.rowcount > 0

    def close(self):
        """Close the database connection."""
        self.conn.close()