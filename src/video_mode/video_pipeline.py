import cv2
import os
from src.object_analysis.yolo_detector import detect_objects_yolo
import easyocr

# Initialize OCR once
reader = easyocr.Reader(['en'])

def process_video(input_path: str, output_path: str = "data/results/output_video.avi"):
    print(f" Processing video: {input_path}")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(" Could not open video.")
        return

    # Get video details
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Run YOLOv8 to detect objects
        detections = detect_objects_yolo(frame)

        # Step 2: Draw boxes and run EasyOCR
        for label, (x1, y1, x2, y2) in detections:
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Crop region of interest
            roi = frame[y1:y2, x1:x2]

            # Run OCR on cropped region
            try:
                ocr_result = reader.readtext(roi, detail=0)
                ocr_text = " ".join(ocr_result).strip()
            except:
                ocr_text = ""

            # Combine label + OCR
            display_text = f"{label} | {ocr_text}" if ocr_text else label
            cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"✅ Processed {frame_count} frames. Output saved to `{output_path}`")
