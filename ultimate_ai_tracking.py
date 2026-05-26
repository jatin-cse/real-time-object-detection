from ultralytics import YOLO
import cv2
import time

# Load advanced YOLO model
model = YOLO("yolo11m.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Better resolution
cap.set(3, 1280)
cap.set(4, 720)

prev_time = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run detection + tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.45,
        iou=0.5
    )

    annotated_frame = results[0].plot()

    # FPS Counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Ultimate AI Detection & Tracking",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()