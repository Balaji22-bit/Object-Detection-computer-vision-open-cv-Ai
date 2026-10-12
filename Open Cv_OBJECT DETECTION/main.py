import os
import cv2
from ultralytics import YOLO
import collections
def open_camera():
    camera_index = 0

    if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
        return cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    return cv2.VideoCapture(camera_index)


def main():
    # Load a pretrained object-detection model
    model = YOLO("yolov8n.pt")

    # Open the default webcam
    cap = open_camera()

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    window_name = "Object Detection and Counting"

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read camera frame.")
                break

            # Detect objects in the current frame
            results = model.predict(
                frame,
                conf=0.35,
                verbose=False
            )

            result = results[0]

            # Draw detection boxes and labels
            output = result.plot()

            # Count detected objects by class in this frame
            counts = collections.Counter()

            if result.boxes is not None:
                for class_id in result.boxes.cls.tolist():
                    class_name = model.names[int(class_id)]
                    counts[class_name] += 1

            total = sum(counts.values())

            # Display total object count
            cv2.putText(
                output,
                f"Total Objects: {total}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            # Display counts by object type
            y = 70
            for name, count in sorted(counts.items()):
                cv2.putText(
                    output,
                    f"{name}: {count}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
                y += 25

            cv2.imshow(window_name, output)

            # Exit with Q, Escape, or the window's X button
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            if cv2.getWindowProperty(
                window_name,
                cv2.WND_PROP_VISIBLE
            ) < 1:
                break

    except KeyboardInterrupt:
        print("\nStopping object detection...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print("Camera released. Program closed.")


if __name__ == "__main__":
    main()