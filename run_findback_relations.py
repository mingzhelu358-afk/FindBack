import json
import sqlite3
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from spatial_reasoning import SpatialReasoner


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "findback.db"
ZONES_PATH = BASE_DIR / "zones.json"
SNAPSHOTS_DIR = BASE_DIR / "snapshots"

# 已经确认Camo Camera是编号1
CAMERA_INDEX = 1

OBJECTS = [
    "keys",
    "wallet",
    "eyeglasses",
    "mobile phone",
    "backpack",
    "cup",
    "book",
    "bottle",
    "laptop",
]

# 这些物品会与电脑比较遮挡关系。电脑本身作为参照物。
RELATION_SUBJECTS = [
    "keys",
    "wallet",
    "eyeglasses",
    "mobile phone",
    "backpack",
    "cup",
    "book",
    "bottle",
]

# 连续识别多少次才确认
STABLE_DETECTIONS = 3

# 同一物品留在同一区域时，每30秒补充一条记录
SAVE_REFRESH_SECONDS = 30

# 遮挡关系连续出现两轮AI检测才正式确认；连续三轮消失则结束。
RELATION_STABLE_DETECTIONS = 2
RELATION_END_DETECTIONS = 3
RELATION_REFRESH_SECONDS = 15

SNAPSHOTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

if not ZONES_PATH.exists():
    raise FileNotFoundError(
        "没有找到zones.json，请先运行区域标定程序"
    )

with open(
    ZONES_PATH,
    "r",
    encoding="utf-8"
) as file:
    zone_config = json.load(file)


def build_polygon(
    zone,
    frame_width,
    frame_height
):
    return np.array(
        [
            [
                int(point[0] * frame_width),
                int(point[1] * frame_height),
            ]
            for point in zone["points"]
        ],
        dtype=np.int32
    )


def determine_location(
    bounding_box,
    frame_shape
):
    frame_height, frame_width = frame_shape[:2]

    x1, y1, x2, y2 = bounding_box

    # 使用物品检测框底部中心作为“落点”
    anchor_x = int((x1 + x2) / 2)
    anchor_y = int(y2)

    sorted_zones = sorted(
        zone_config["zones"],
        key=lambda zone: zone.get(
            "priority",
            0
        ),
        reverse=True
    )

    for zone_number, zone in enumerate(
        sorted_zones,
        start=1
    ):
        polygon = build_polygon(
            zone,
            frame_width,
            frame_height
        )

        result = cv2.pointPolygonTest(
            polygon,
            (anchor_x, anchor_y),
            False
        )

        if result >= 0:
            return (
                zone["name"],
                zone_number,
                (anchor_x, anchor_y),
            )

    return (
        "未知区域",
        0,
        (anchor_x, anchor_y),
    )


connection = sqlite3.connect(
    DATABASE_PATH
)

# 即使已经运行过init_db.py，
# 这里仍然检查一次，避免表格缺失
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

connection.execute(
    """
    CREATE TABLE IF NOT EXISTS spatial_relations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_name TEXT NOT NULL,
        relation TEXT NOT NULL,
        reference_name TEXT NOT NULL,
        location TEXT NOT NULL,
        confidence REAL NOT NULL,
        visible_ratio REAL NOT NULL,
        status TEXT NOT NULL,
        detected_at TEXT NOT NULL,
        evidence TEXT,
        snapshot_path TEXT
    );
    """
)

# 上一次程序如果被直接关闭，数据库里可能还留有active状态。
# 新一轮运行会重新观察并重新建立当前有效关系。
connection.execute(
    """
    UPDATE spatial_relations
    SET status = 'ended'
    WHERE status = 'active';
    """
)

connection.commit()

device = (
    0
    if torch.cuda.is_available()
    else "cpu"
)

print("AI运行设备：", device)

model = YOLO(
    "yolov8s-world.pt"
)

model.set_classes(
    OBJECTS
)

camera = None
first_valid_frame = None
selected_backend_name = None

# Windows上的Camo有时只有MSMF或DirectShow中的一个能够正常输出。
# 优先尝试当前实测正常的MSMF，失败后自动切换到DirectShow。
camera_backends = [
    ("MSMF", cv2.CAP_MSMF),
    ("DirectShow", cv2.CAP_DSHOW),
]

for backend_name, backend_code in camera_backends:
    print(
        f"正在尝试Camo Camera接口："
        f"{backend_name}……"
    )

    candidate_camera = cv2.VideoCapture(
        CAMERA_INDEX,
        backend_code,
    )

    candidate_camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )
    candidate_camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )
    candidate_camera.set(
        cv2.CAP_PROP_FPS,
        15,
    )
    candidate_camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    if not candidate_camera.isOpened():
        candidate_camera.release()
        print(f"{backend_name}无法打开")
        continue

    warmup_deadline = time.monotonic() + 15

    while time.monotonic() < warmup_deadline:
        success, candidate_frame = (
            candidate_camera.read()
        )

        if (
            success
            and candidate_frame is not None
            and candidate_frame.size > 0
            and float(candidate_frame.mean()) > 5.0
        ):
            camera = candidate_camera
            first_valid_frame = candidate_frame
            selected_backend_name = backend_name
            break

        time.sleep(0.05)

    if first_valid_frame is not None:
        break

    candidate_camera.release()
    print(f"{backend_name}只返回黑色画面，尝试下一个接口")

if camera is None or first_valid_frame is None:
    connection.close()
    raise RuntimeError(
        "Camo Camera的MSMF和DirectShow接口都没有正常画面。"
        "请确认Camo Studio能看到iPhone实时画面。"
    )

print(
    f"Camo Camera画面准备完成（{selected_backend_name}）："
    f"{first_valid_frame.shape[1]}x{first_valid_frame.shape[0]}，"
    f"平均亮度{first_valid_frame.mean():.1f}"
)

# 保存连续检测状态
stable_states = defaultdict(
    lambda: {
        "location": None,
        "count": 0,
    }
)

# 保存最后一次写入数据库的状态
last_saved = {}

# 单摄像头空间关系推理器及其稳定状态。
spatial_reasoner = SpatialReasoner(
    subjects=RELATION_SUBJECTS,
    reference_name="laptop",
    memory_seconds=90,
)

relation_stability = defaultdict(int)
relation_missing_counts = defaultdict(int)
active_relations = {}

print("\n区域对应关系：")

for index, zone in enumerate(
    zone_config["zones"],
    start=1
):
    print(
        f"Z{index} = {zone['name']}"
    )

print("\nFindBack开始运行")
print("按Q退出\n")

try:
    pending_frame = first_valid_frame
    last_black_frame_warning = 0.0

    while True:
        if pending_frame is not None:
            success = True
            frame = pending_frame
            pending_frame = None
        else:
            success, frame = camera.read()

        if (
            not success
            or frame is None
            or frame.size == 0
            or float(frame.mean()) <= 5.0
        ):
            current_time = time.monotonic()

            if (
                current_time
                - last_black_frame_warning
                >= 3
            ):
                print(
                    "暂时收到黑色画面，"
                    "正在等待Camo恢复……"
                )
                last_black_frame_warning = (
                    current_time
                )

            # 即使画面暂时中断，也允许用户按Q退出。
            if (
                cv2.waitKey(20) & 0xFF
                == ord("q")
            ):
                break

            time.sleep(0.05)
            continue

        display_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]

        # 在画面中绘制区域边界
        for zone_index, zone in enumerate(
            zone_config["zones"],
            start=1
        ):
            polygon = build_polygon(
                zone,
                frame_width,
                frame_height
            )

            cv2.polylines(
                display_frame,
                [polygon],
                True,
                (255, 255, 0),
                2
            )

            first_x, first_y = polygon[0]

            cv2.putText(
                display_frame,
                f"Z{zone_index}",
                (int(first_x), int(first_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

        results = model.predict(
            frame,
            conf=0.20,
            imgsz=416,
            device=device,
            verbose=False
        )

        result = results[0]
        current_detections = {}

        if result.boxes is not None:
            for box in result.boxes:
                confidence = float(
                    box.conf[0]
                )

                class_id = int(
                    box.cls[0]
                )

                object_name = result.names[
                    class_id
                ]

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .tolist()
                )

                (
                    location,
                    zone_number,
                    anchor_point,
                ) = determine_location(
                    (x1, y1, x2, y2),
                    frame.shape
                )

                # 同类物品只保留置信度最高的一个
                previous = current_detections.get(
                    object_name
                )

                if (
                    previous is None
                    or confidence
                    > previous["confidence"]
                ):
                    current_detections[
                        object_name
                    ] = {
                        "confidence": confidence,
                        "location": location,
                        "zone_number": zone_number,
                        "anchor": anchor_point,
                        "box": (
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2),
                        ),
                    }

        # 未检测到的类别清空连续计数
        for object_name in OBJECTS:
            if object_name not in current_detections:
                stable_states[
                    object_name
                ]["count"] = 0

        for (
            object_name,
            detection,
        ) in current_detections.items():

            location = detection["location"]
            confidence = detection["confidence"]
            zone_number = detection["zone_number"]
            anchor_x, anchor_y = detection["anchor"]
            x1, y1, x2, y2 = detection["box"]

            state = stable_states[
                object_name
            ]

            if state["location"] == location:
                state["count"] += 1
            else:
                state["location"] = location
                state["count"] = 1

            # 绘制检测框
            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.circle(
                display_frame,
                (anchor_x, anchor_y),
                6,
                (0, 0, 255),
                -1
            )

            label = (
                f"{object_name} "
                f"{confidence:.2f} "
                f"Z{zone_number}"
            )

            cv2.putText(
                display_frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            if (
                state["count"]
                < STABLE_DETECTIONS
            ):
                continue

            now = datetime.now()
            previous_save = last_saved.get(
                object_name
            )

            should_save = (
                previous_save is None
                or previous_save["location"]
                != location
                or (
                    now - previous_save["time"]
                ).total_seconds()
                >= SAVE_REFRESH_SECONDS
            )

            if not should_save:
                continue

            timestamp_for_file = now.strftime(
                "%Y%m%d_%H%M%S"
            )

            safe_name = object_name.replace(
                " ",
                "_"
            )

            snapshot_path = (
                SNAPSHOTS_DIR
                / (
                    f"{safe_name}_"
                    f"{timestamp_for_file}.jpg"
                )
            )

            cv2.imwrite(
                str(snapshot_path),
                display_frame
            )

            detected_at = now.isoformat(
                timespec="seconds"
            )

            connection.execute(
                """
                INSERT INTO observations (
                    object_name,
                    location,
                    confidence,
                    detected_at,
                    snapshot_path
                )
                VALUES (?, ?, ?, ?, ?);
                """,
                (
                    object_name,
                    location,
                    confidence,
                    detected_at,
                    str(snapshot_path),
                )
            )

            connection.commit()

            last_saved[object_name] = {
                "location": location,
                "time": now,
            }

            print(
                f"[{detected_at}] "
                f"{object_name} → "
                f"{location} "
                f"({confidence:.2f})"
            )

        # 根据当前画面和此前清晰画面推断遮挡关系。
        relations = spatial_reasoner.update(
            current_detections
        )
        current_relation_keys = set()

        for relation in relations:
            relation_key = (
                relation["subject_name"],
                relation["relation"],
                relation["reference_name"],
            )
            current_relation_keys.add(relation_key)
            relation_missing_counts[relation_key] = 0
            relation_stability[relation_key] += 1

            subject_x1, subject_y1, subject_x2, subject_y2 = (
                relation["subject_box"]
            )

            # 橙色框表示系统推测的被遮挡物品位置。
            cv2.rectangle(
                display_frame,
                (subject_x1, subject_y1),
                (subject_x2, subject_y2),
                (0, 165, 255),
                2,
            )

            relation_label = (
                f"{relation['subject_name']} likely behind "
                f"{relation['reference_name']} "
                f"{relation['confidence']:.2f}"
            )

            cv2.putText(
                display_frame,
                relation_label,
                (
                    subject_x1,
                    max(50, subject_y1 - 32),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 165, 255),
                2,
            )

            if (
                relation_stability[relation_key]
                < RELATION_STABLE_DETECTIONS
            ):
                continue

            now = datetime.now()
            active_state = active_relations.get(
                relation_key
            )

            should_save_relation = (
                active_state is None
                or (
                    now - active_state["last_saved"]
                ).total_seconds()
                >= RELATION_REFRESH_SECONDS
            )

            if not should_save_relation:
                active_state["data"] = relation
                continue

            timestamp_for_file = now.strftime(
                "%Y%m%d_%H%M%S"
            )
            relation_snapshot_path = (
                SNAPSHOTS_DIR
                / (
                    "relation_"
                    f"{relation['subject_name'].replace(' ', '_')}_"
                    f"{timestamp_for_file}.jpg"
                )
            )

            cv2.imwrite(
                str(relation_snapshot_path),
                display_frame,
            )

            detected_at = now.isoformat(
                timespec="seconds"
            )

            connection.execute(
                """
                INSERT INTO spatial_relations (
                    subject_name,
                    relation,
                    reference_name,
                    location,
                    confidence,
                    visible_ratio,
                    status,
                    detected_at,
                    evidence,
                    snapshot_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    relation["subject_name"],
                    relation["relation"],
                    relation["reference_name"],
                    relation["location"],
                    relation["confidence"],
                    relation["visible_ratio"],
                    "active",
                    detected_at,
                    relation["evidence"],
                    str(relation_snapshot_path),
                ),
            )
            connection.commit()

            active_relations[relation_key] = {
                "data": relation,
                "last_saved": now,
            }

            print(
                f"[{detected_at}] 空间关系："
                f"{relation['subject_name']} "
                f"可能在{relation['reference_name']}后面 "
                f"({relation['confidence']:.2f})"
            )

        # 当前未再出现的候选关系，清空确认计数。
        for relation_key in list(
            relation_stability
        ):
            if relation_key not in current_relation_keys:
                relation_stability[relation_key] = 0

        # 已确认的关系连续多轮消失后，写入ended事件，避免查询旧关系。
        for relation_key in list(active_relations):
            if relation_key in current_relation_keys:
                continue

            relation_missing_counts[relation_key] += 1

            if (
                relation_missing_counts[relation_key]
                < RELATION_END_DETECTIONS
            ):
                continue

            now = datetime.now()
            detected_at = now.isoformat(
                timespec="seconds"
            )
            last_relation = active_relations[
                relation_key
            ]["data"]

            connection.execute(
                """
                INSERT INTO spatial_relations (
                    subject_name,
                    relation,
                    reference_name,
                    location,
                    confidence,
                    visible_ratio,
                    status,
                    detected_at,
                    evidence,
                    snapshot_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    last_relation["subject_name"],
                    last_relation["relation"],
                    last_relation["reference_name"],
                    last_relation["location"],
                    last_relation["confidence"],
                    last_relation["visible_ratio"],
                    "ended",
                    detected_at,
                    "当前画面不再支持这一遮挡关系",
                    None,
                ),
            )
            connection.commit()

            print(
                f"[{detected_at}] 空间关系结束："
                f"{last_relation['subject_name']} / "
                f"{last_relation['reference_name']}"
            )

            del active_relations[relation_key]
            relation_missing_counts.pop(
                relation_key,
                None,
            )

        cv2.imshow(
            "FindBack",
            display_frame
        )

        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):
            break

finally:
    camera.release()
    connection.close()
    cv2.destroyAllWindows()
