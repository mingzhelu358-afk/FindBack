import cv2

# 编号1很可能是Camo Camera
CAMERA_INDEX = 1

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_MSMF
)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camera.set(cv2.CAP_PROP_FPS, 15)

if not camera.isOpened():
    raise RuntimeError("无法打开摄像头")

while True:
    success, frame = camera.read()

    if not success:
        print("无法读取摄像头画面")
        break

    cv2.putText(
        frame,
        f"Camera Index: {CAMERA_INDEX}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "iPhone Camera Test",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
