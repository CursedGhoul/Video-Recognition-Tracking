# pip install opencv-python
# pip install numpy
# pip install ultralytics
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8x model
model = YOLO('yolov8x.pt') 

def process_frame(frame):
    # Convert frame to RGB (required by YOLOv8)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform inference
    results = model(rgb_frame)
    
    # Extract predictions
    predictions = results[0].boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2, confidence, class)
    confidences = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    
    detected_objects = []
    for box, conf, cls in zip(predictions, confidences, classes):
        label = model.names[int(cls)]
        detected_objects.append(label)
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame, detected_objects

def detect_anomalies(video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    anomaly_log = []

    # Get the width, height, and FPS of the video frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Default value if FPS is not obtained correctly

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        processed_frame, detected_objects = process_frame(frame)

        # Simple anomaly detection: if a specific object is detected, log it
        anomalies = [obj for obj in detected_objects if obj in ["person", "car"]]  # Example anomalies
        if anomalies:
            anomaly_log.append((frame_count, anomalies))
            out.write(processed_frame)  # Save the frame to the output video

        # Display the frame with detected objects
        cv2.imshow("Frame", processed_frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  # Release the VideoWriter object
    cv2.destroyAllWindows()

    # Log the anomalies
    with open("anomaly_log.txt", "w") as f:
        for log in anomaly_log:
            f.write(f"Frame {log[0]}: Detected anomalies - {', '.join(log[1])}\n")

    print("Anomaly detection complete. Check anomaly_log.txt for details.")
    print(f"Saved video with detected anomalies to {output_video_path}")

# Example usage
detect_anomalies("Path To Input video", "{path to OUTPUT VIDEO}")
