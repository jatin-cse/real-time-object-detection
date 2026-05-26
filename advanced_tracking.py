from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

# Lightweight YOLO model
model = YOLO("yolov8n.pt")

tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture(0)

# Lower resolution
cap.set(3, 640)
cap.set(4, 480)

frame_skip = 2
frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Skip frames
    if frame_count % frame_skip != 0:
        continue

    results = model(frame)

    detections = []

    for result in results:

        boxes = result.boxes

        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]

            x1, y1, x2, y2 = map(int,
                                 [x1, y1, x2, y2])

            conf = float(box.conf[0])
            cls = int(box.cls[0])

            class_name = model.names[cls]

            if conf > 0.5:

                detections.append(
                    ([x1, y1,
                      x2 - x1,
                      y2 - y1],
                     conf,
                     class_name)
                )

    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = map(int, ltrb)

        class_name = track.get_det_class()

        cv2.rectangle(frame,
                      (x1, y1),
                      (x2, y2),
                      (0, 255, 0),
                      2)

        cv2.putText(frame,
                    f"{class_name} ID:{track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

    cv2.imshow("Optimized Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()