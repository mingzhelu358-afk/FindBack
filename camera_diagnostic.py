import sys
import time
from pathlib import Path

import cv2


OUTPUT_DIR = Path(sys.argv[1]).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tests = [
    (0, cv2.CAP_DSHOW, "dshow"),
    (1, cv2.CAP_DSHOW, "dshow"),
    (1, cv2.CAP_MSMF, "msmf"),
]

for camera_index, backend, backend_name in tests:
    camera = cv2.VideoCapture(camera_index, backend)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 15)

    opened = camera.isOpened()
    successes = 0
    brightest_mean = -1.0
    brightest_frame = None

    if opened:
        for _ in range(50):
            success, frame = camera.read()
            if success and frame is not None and frame.size:
                successes += 1
                frame_mean = float(frame.mean())
                if frame_mean > brightest_mean:
                    brightest_mean = frame_mean
                    brightest_frame = frame.copy()
            time.sleep(0.04)

    camera.release()

    output_path = OUTPUT_DIR / f"camera_{camera_index}_{backend_name}.jpg"
    if brightest_frame is not None:
        cv2.imwrite(str(output_path), brightest_frame)

    print(
        f"index={camera_index} backend={backend_name} "
        f"opened={opened} reads={successes}/50 "
        f"brightest_mean={brightest_mean:.2f} "
        f"saved={output_path if brightest_frame is not None else 'none'}"
    )
