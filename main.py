import os
import time
import cv2
from ultralytics import YOLO

# Create output folder
os.makedirs("output", exist_ok=True)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

# Get webcam resolution
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Webcam resolution:", frame_width, "x", frame_height)

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*"XVID")

out = cv2.VideoWriter(
    "output/demo.avi",
    fourcc,
    20.0,
    (frame_width, frame_height)
)

if not out.isOpened():
    print("❌ Video writer failed to open.")
    cap.release()
    exit()

print("✅ Video recording started.")

prev_time = time.time()
screenshot_count = 1

while True:

    # Read frame
    success, frame = cap.read()

    if not success:
        print("❌ Failed to read webcam frame.")
        break

    # YOLO detection
    results = model(frame)

    # Object counter
    object_count = {}

    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            object_name = model.names[class_id]

            confidence = round(float(box.conf[0]) * 100, 2)

            if object_name in object_count:
                object_count[object_name] += 1
            else:
                object_count[object_name] = 1

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Calculate FPS
    current_time = time.time()
    time_difference = current_time - prev_time

    if time_difference > 0:
        fps = int(1 / time_difference)
    else:
        fps = 0

    prev_time = current_time

    # Display FPS
    cv2.putText(
        annotated_frame,
        f"FPS: {fps}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    # Display object counts
    y = 70

    for object_name, count in object_count.items():

        cv2.putText(
            annotated_frame,
            f"{object_name}: {count}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        y += 30

    # Show video
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    # Record video
    out.write(annotated_frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF

    # Save screenshot
    if key == ord("s"):

        filename = os.path.join(
            "output",
            f"screenshot_{screenshot_count}.jpg"
        )

        saved = cv2.imwrite(filename, annotated_frame)

        if saved:
            print(f"✅ Screenshot saved: {filename}")
            screenshot_count += 1
        else:
            print("❌ Screenshot failed.")

    # Quit
    elif key == ord("q"):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Program finished.")
print("🎥 Video saved as: output/demo.avi")