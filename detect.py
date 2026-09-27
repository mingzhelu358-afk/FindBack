import cv2
import torch
from ultralytics import YOLO

CAMERA_INDEX = 1  # 改成Camo Camera编号

OBJECTS = [
    "keys",
    "wallet",
    "eyeglasses",
    "mobile phone",
    "backpack",
    "cup",
    "book",
    "bottle",
]

device = 0 if torch.cuda.is_available() else "cpu"
print("AI运行设备：", device)

model = YOLO("yolov8s-world.pt")
model.set_classes(OBJECTS)

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camera.set(cv2.CAP_PROP_FPS, 15)

while True:
    success, frame = camera.read()

    if not success:
        print("摄像头断开")
        break

    results = model.predict(
        frame,
        conf=0.20,
        imgsz=640,
        device=device,
        verbose=False,
    )

    result = results[0]
    annotated_frame = result.plot()

    cv2.imshow(
        "FindBack Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()