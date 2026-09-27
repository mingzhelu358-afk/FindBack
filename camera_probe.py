import cv2

for camera_index in range(10):
    camera = cv2.VideoCapture(
        camera_index,
        cv2.CAP_DSHOW
    )

    success, frame = camera.read()

    if success:
        print(
            f"摄像头编号 {camera_index} 可用，"
            f"画面尺寸：{frame.shape}"
        )

    camera.release()