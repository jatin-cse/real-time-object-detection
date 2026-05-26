from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8m.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Tracking
    results = model.track(frame, persist=True)

    # Draw tracking results
    annotated_frame = results[0].plot()

    # Show frame
    cv2.imshow("Object Tracking", annotated_frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()