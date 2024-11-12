import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import sys
import os

def main():
    # Initialize the YOLO model with ByteTrack
    try:
        # Initialize YOLO with the desired model weights
        # Ensure 'yolov8_custom.pt' is in the current directory or provide the correct path
        model = YOLO("yolov8_custom.pt")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        sys.exit(1)

    # Prompt the user to input the video file path
    video_path = input("Please enter the local video file path: ").strip()

    # Check if the video file exists
    if not os.path.isfile(video_path):
        print(f"Error: Video file '{video_path}' does not exist.")
        sys.exit(1)

    # Attempt to open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'. Please check the path and try again.")
        sys.exit(1)

    # Initialize a dictionary to store tracking history for each object ID
    track_history = defaultdict(list)

    # Define classes to ignore
    classes_to_ignore = ['goalie', 'referee']  # Replace with actual class names
    # If you have class IDs, you can use them instead. For example:
    # classes_to_ignore = [0, 1]  # Replace with actual class IDs

    # Retrieve class names from the model
    try:
        model_classes = model.names  # Dictionary mapping class IDs to class names
    except AttributeError:
        print("Error: Unable to retrieve class names from the model.")
        sys.exit(1)

    # Map class names to class IDs for easier filtering
    classes_to_ignore_ids = []
    for class_id, class_name in model_classes.items():
        if class_name.lower() in [cls.lower() for cls in classes_to_ignore]:
            classes_to_ignore_ids.append(class_id)

    print("Processing video. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("End of video or cannot read the frame.")
            break

        # Perform object tracking on the current frame using ByteTrack
        # Ensure 'botsort.yaml' is in the current directory or provide the correct path
        try:
            results = model.track(frame, persist=True, tracker="botsort.yaml")
        except Exception as e:
            print(f"Error during tracking: {e}")
            break

        # Extract relevant data from the results
        boxes = results[0].boxes  # Box object containing all detections
        if boxes is None:
            # No detections in this frame
            annotated_frame = frame.copy()
            cv2.imshow("YOLO11 Tracking with ByteTrack", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quitting video processing.")
                break
            continue

        # Create an annotated frame with bounding boxes and tracking information
        annotated_frame = frame.copy()

        for box in boxes:
            cls_id = int(box.cls.cpu().numpy())  # Class ID
            if cls_id in classes_to_ignore_ids:
                continue  # Skip ignored classes

            track_id = int(box.id.cpu().numpy()) if box.id is not None else None
            if track_id is None:
                continue  # Skip if no track ID

            # Bounding box coordinates
            x, y, w, h = box.xywh.cpu().numpy()[0]  # [x_center, y_center, width, height]
            # Convert from center coordinates to top-left corner
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            # Update the tracking history for the current object ID
            track_history[track_id].append((int(x), int(y)))

            # Maintain only the last 30 positions for each track
            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)

            # Prepare points for drawing the tracking polyline
            points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))

            # Draw the bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw the tracking polyline on the annotated frame
            cv2.polylines(
                annotated_frame,
                [points],
                isClosed=False,
                color=(0, 255, 0),  # Green color for tracking lines
                thickness=2
            )

            # Add only the track ID near the bounding box
            font_scale = 0.5  # Smaller font size
            thickness = 1  # Thinner text
            text = f'ID: {track_id}'
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_w, text_h = text_size
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - text_h - 4),
                (x1 + text_w + 4, y1),
                (0, 255, 0),
                -1
            )
            cv2.putText(
                annotated_frame,
                text,
                (x1 + 2, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),  # Black text color for better visibility
                thickness,
                cv2.LINE_AA
            )

        # Display the annotated frame in a window
        cv2.imshow("YOLO11 Tracking with ByteTrack", annotated_frame)

        # Exit the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quitting video processing.")
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
