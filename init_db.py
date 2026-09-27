import sqlite3
from pathlib import Path

database_path = (
    Path(__file__).resolve().parent
    / "findback.db"
)

with sqlite3.connect(database_path) as connection:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            object_name TEXT NOT NULL,
            location TEXT NOT NULL,
            confidence REAL NOT NULL,
            detected_at TEXT NOT NULL,
            snapshot_path TEXT
        );
        """
    )

    connection.commit()

    tables = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table';
        """
    ).fetchall()

print(f"数据库已创建：{database_path}")
print(f"数据表：{tables}")