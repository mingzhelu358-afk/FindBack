import sqlite3
import threading
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "findback.db"
INDEX_PATH = BASE_DIR / "templates" / "index.html"
SNAPSHOTS_DIR = (BASE_DIR / "snapshots").resolve()

OBJECT_LABELS = {
    "keys": "钥匙",
    "wallet": "钱包",
    "eyeglasses": "眼镜",
    "mobile phone": "手机",
    "backpack": "背包",
    "cup": "杯子",
    "book": "书",
    "bottle": "瓶子",
    "laptop": "电脑",
}

ALIASES = {
    "钥匙": "keys",
    "key": "keys",
    "keys": "keys",
    "钱包": "wallet",
    "wallet": "wallet",
    "眼镜": "eyeglasses",
    "眼睛": "eyeglasses",
    "eyeglasses": "eyeglasses",
    "glasses": "eyeglasses",
    "手机": "mobile phone",
    "电话": "mobile phone",
    "iphone": "mobile phone",
    "mobile phone": "mobile phone",
    "phone": "mobile phone",
    "背包": "backpack",
    "书包": "backpack",
    "backpack": "backpack",
    "杯子": "cup",
    "水杯": "cup",
    "马克杯": "cup",
    "cup": "cup",
    "书本": "book",
    "书": "book",
    "book": "book",
    "水瓶": "bottle",
    "瓶子": "bottle",
    "bottle": "bottle",
    "笔记本电脑": "laptop",
    "笔记本": "laptop",
    "电脑": "laptop",
    "laptop": "laptop",
}

app = FastAPI(title="FindBack 查询助手")


def get_connection():
    if not DATABASE_PATH.exists():
        raise HTTPException(status_code=503, detail="还没有找到 findback.db 数据库")

    connection = sqlite3.connect(DATABASE_PATH, timeout=5)
    connection.row_factory = sqlite3.Row
    return connection


def identify_object(question: str):
    normalized = question.strip().lower()

    # 优先匹配较长的别名，避免“水杯”先被“杯”一类短词截断。
    for alias in sorted(ALIASES, key=len, reverse=True):
        if alias.lower() in normalized:
            return ALIASES[alias]

    return None


def serialize_row(row):
    object_name = row["object_name"]
    return {
        "id": row["id"],
        "object_name": object_name,
        "object_label": OBJECT_LABELS.get(object_name, object_name),
        "location": row["location"],
        "confidence": round(float(row["confidence"]), 3),
        "detected_at": row["detected_at"],
        "snapshot_url": f"/api/snapshot/{row['id']}" if row["snapshot_path"] else None,
    }


def get_current_relation(connection, subject_name: str):
    table_exists = connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = 'spatial_relations';
        """
    ).fetchone()

    if table_exists is None:
        return None

    row = connection.execute(
        """
        SELECT id, subject_name, relation, reference_name, location,
               confidence, visible_ratio, status, detected_at,
               evidence, snapshot_path
        FROM spatial_relations
        WHERE subject_name = ?
        ORDER BY id DESC
        LIMIT 1;
        """,
        (subject_name,),
    ).fetchone()

    if row is None or row["status"] != "active":
        return None

    return row


@app.get("/", response_class=HTMLResponse)
def index():
    if not INDEX_PATH.exists():
        raise HTTPException(status_code=500, detail="缺少 templates/index.html")
    return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))


@app.get("/api/query")
def query_object(q: str = Query(min_length=1, max_length=100)):
    object_name = identify_object(q)

    if object_name is None:
        choices = "、".join(OBJECT_LABELS.values())
        return {
            "found": False,
            "recognized": False,
            "answer": f"我还没听出你要找什么。现在可以查询：{choices}。",
        }

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, object_name, location, confidence, detected_at, snapshot_path
            FROM observations
            WHERE object_name = ?
            ORDER BY id DESC
            LIMIT 1;
            """,
            (object_name,),
        ).fetchone()
        relation_row = get_current_relation(
            connection,
            object_name,
        )

    label = OBJECT_LABELS.get(object_name, object_name)

    if row is None:
        return {
            "found": False,
            "recognized": True,
            "object_name": object_name,
            "object_label": label,
            "answer": f"我暂时还没有看到{label}。请让摄像头继续观察一会儿。",
        }

    observation = serialize_row(row)

    if relation_row is not None:
        reference_name = relation_row[
            "reference_name"
        ]
        reference_label = OBJECT_LABELS.get(
            reference_name,
            reference_name,
        )
        visible_percent = round(
            float(relation_row["visible_ratio"])
            * 100
        )

        if visible_percent <= 1:
            visibility_text = "目前已经完全被遮挡"
        else:
            visibility_text = (
                f"目前估计还能看到{visible_percent}%"
            )

        observation["relation"] = {
            "type": relation_row["relation"],
            "reference_name": reference_name,
            "reference_label": reference_label,
            "confidence": round(
                float(relation_row["confidence"]),
                3,
            ),
            "visible_ratio": round(
                float(relation_row["visible_ratio"]),
                3,
            ),
            "detected_at": relation_row[
                "detected_at"
            ],
            "evidence": (
                relation_row["evidence"] or ""
            ).replace(
                reference_name,
                reference_label,
            ),
        }

        if relation_row["snapshot_path"]:
            observation["snapshot_url"] = (
                "/api/relation-snapshot/"
                f"{relation_row['id']}"
            )

        answer = (
            f"{label}最后一次出现在{row['location']}，"
            f"可能在{reference_label}后面，"
            f"{visibility_text}。"
        )
    else:
        answer = (
            f"{label}最后一次出现在"
            f"{row['location']}。"
        )

    observation.update(
        {
            "found": True,
            "recognized": True,
            "answer": answer,
        }
    )
    return observation


@app.get("/api/recent")
def recent_objects():
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT o.id, o.object_name, o.location, o.confidence,
                   o.detected_at, o.snapshot_path
            FROM observations AS o
            INNER JOIN (
                SELECT object_name, MAX(id) AS newest_id
                FROM observations
                GROUP BY object_name
            ) AS newest
            ON o.id = newest.newest_id
            ORDER BY o.id DESC
            LIMIT 12;
            """
        ).fetchall()

    return {"items": [serialize_row(row) for row in rows]}


@app.get("/api/snapshot/{observation_id}")
def snapshot(observation_id: int):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT snapshot_path FROM observations WHERE id = ?;",
            (observation_id,),
        ).fetchone()

    if row is None or not row["snapshot_path"]:
        raise HTTPException(status_code=404, detail="没有找到这张快照")

    return serve_snapshot(row["snapshot_path"])


@app.get("/api/relation-snapshot/{relation_id}")
def relation_snapshot(relation_id: int):
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT snapshot_path
            FROM spatial_relations
            WHERE id = ?;
            """,
            (relation_id,),
        ).fetchone()

    if row is None or not row["snapshot_path"]:
        raise HTTPException(
            status_code=404,
            detail="没有找到这张关系快照",
        )

    return serve_snapshot(row["snapshot_path"])


def serve_snapshot(raw_path: str):
    snapshot_path = Path(raw_path)
    if not snapshot_path.is_absolute():
        snapshot_path = BASE_DIR / snapshot_path
    snapshot_path = snapshot_path.resolve()

    try:
        snapshot_path.relative_to(SNAPSHOTS_DIR)
    except ValueError as error:
        raise HTTPException(status_code=403, detail="快照路径无效") from error

    if not snapshot_path.is_file():
        raise HTTPException(status_code=404, detail="快照文件不存在")

    return FileResponse(snapshot_path, media_type="image/jpeg")


if __name__ == "__main__":
    url = "http://127.0.0.1:8000"
    print("\nFindBack 查询页面正在启动")
    print(f"如果浏览器没有自动打开，请访问：{url}")
    print("按 Ctrl+C 停止查询服务\n")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
