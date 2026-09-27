import sqlite3

connection = sqlite3.connect(
    "findback.db"
)

rows = connection.execute(
    """
    SELECT
        object_name,
        location,
        confidence,
        detected_at,
        snapshot_path
    FROM observations
    ORDER BY id DESC
    LIMIT 20;
    """
).fetchall()

connection.close()

if not rows:
    print("数据库中还没有记录")
else:
    for row in rows:
        print(row)