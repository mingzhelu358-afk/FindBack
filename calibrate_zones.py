import json
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

# Camo Camera的编号
CAMERA_INDEX = 1

# 按照这个顺序逐一圈选
ZONE_NAMES = [
    "书桌左侧",
    "书桌中间",
    "书桌右侧",
    "书桌下方地面",
]

COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 165, 255),
    (255, 0, 255),
]

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_MSMF
)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)

camera.set(
    cv2.CAP_PROP_FPS,
    15
)

print("\n正在打开摄像头预览……")
print("看到正常画面后按S截取标定画面")
print("按Q取消\n")

frame = None

while True:
    success, live_frame = camera.read()

    if not success:
        print("暂时无法读取摄像头，正在重试")
        continue

    cv2.putText(
        live_frame,
        "Press S to capture",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Camera Preview",
        live_frame
    )

    key = cv2.waitKey(20) & 0xFF

    if key == ord("s"):
        frame = live_frame.copy()
        print("已经截取标定画面")
        break

    if key == ord("q"):
        camera.release()
        cv2.destroyAllWindows()
        print("已取消")
        raise SystemExit

camera.release()
cv2.destroyAllWindows()

if frame is None:
    raise RuntimeError("没有获得标定画面")
frame_height, frame_width = frame.shape[:2]

zones = []
current_points = []


def mouse_callback(
    event,
    x,
    y,
    flags,
    parameter
):
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append([x, y])
        print(f"添加顶点：({x}, {y})")


cv2.namedWindow(
    "Zone Calibration",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Zone Calibration",
    frame_width,
    frame_height
)

cv2.setMouseCallback(
    "Zone Calibration",
    mouse_callback
)

print("\n区域顺序：")

for index, name in enumerate(
    ZONE_NAMES,
    start=1
):
    print(f"Z{index} = {name}")

print("\n操作方法：")
print("鼠标左键：添加边界点")
print("Enter：完成当前区域")
print("U：撤销最后一个点")
print("R：重新绘制当前区域")
print("Q：退出且不保存\n")

while len(zones) < len(ZONE_NAMES):
    canvas = frame.copy()
    overlay = frame.copy()

    # 绘制已经完成的区域
    for index, zone in enumerate(zones):
        color = COLORS[
            index % len(COLORS)
        ]

        polygon = np.array(
            zone["points_pixels"],
            dtype=np.int32
        )

        cv2.fillPoly(
            overlay,
            [polygon],
            color
        )

    canvas = cv2.addWeighted(
        overlay,
        0.25,
        canvas,
        0.75,
        0
    )

    # 绘制已完成区域的边界
    for index, zone in enumerate(zones):
        color = COLORS[
            index % len(COLORS)
        ]

        polygon = np.array(
            zone["points_pixels"],
            dtype=np.int32
        )

        cv2.polylines(
            canvas,
            [polygon],
            True,
            color,
            3
        )

        first_x, first_y = polygon[0]

        cv2.putText(
            canvas,
            f"Z{index + 1}",
            (
                int(first_x),
                int(first_y)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    # 绘制当前正在标记的点
    if current_points:
        current_array = np.array(
            current_points,
            dtype=np.int32
        )

        for point_x, point_y in current_points:
            cv2.circle(
                canvas,
                (point_x, point_y),
                5,
                (0, 0, 255),
                -1
            )

        if len(current_points) >= 2:
            cv2.polylines(
                canvas,
                [current_array],
                False,
                (0, 0, 255),
                2
            )

    current_zone_number = len(zones) + 1

    cv2.putText(
        canvas,
        f"Drawing Z{current_zone_number}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    cv2.imshow(
        "Zone Calibration",
        canvas
    )

    key = cv2.waitKey(20) & 0xFF

    # Enter
    if key in (10, 13):
        if len(current_points) < 3:
            print("当前区域至少需要三个顶点")
            continue

        zone_name = ZONE_NAMES[
            len(zones)
        ]

        zones.append({
            "name": zone_name,
            "priority": 100 - len(zones),
            "points_pixels": (
                current_points.copy()
            ),
        })

        current_points.clear()

        print(f"已完成：{zone_name}")

        if len(zones) < len(ZONE_NAMES):
            print(
                "下一个区域："
                + ZONE_NAMES[len(zones)]
            )

    # 撤销一个点
    elif key == ord("u"):
        if current_points:
            current_points.pop()

    # 重新画当前区域
    elif key == ord("r"):
        current_points.clear()

    # 取消
    elif key == ord("q"):
        cv2.destroyAllWindows()
        print("已取消标定")
        raise SystemExit

cv2.destroyAllWindows()

output_zones = []

for zone in zones:
    normalized_points = []

    for point_x, point_y in zone[
        "points_pixels"
    ]:
        normalized_points.append([
            round(
                point_x / frame_width,
                6
            ),
            round(
                point_y / frame_height,
                6
            ),
        ])

    output_zones.append({
        "name": zone["name"],
        "priority": zone["priority"],
        "points": normalized_points,
    })

output = {
    "image_width": frame_width,
    "image_height": frame_height,
    "zones": output_zones,
}

zones_path = BASE_DIR / "zones.json"

with open(
    zones_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        output,
        file,
        ensure_ascii=False,
        indent=2
    )

# 保存原始标定画面
cv2.imwrite(
    str(BASE_DIR / "calibration_frame.jpg"),
    frame
)

print("\n标定完成")
print(f"区域文件：{zones_path}")
print("现在可以运行run_findback.py")
