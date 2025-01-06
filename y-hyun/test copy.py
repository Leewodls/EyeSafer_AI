import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(r"C:\Users\joyon\EyeSafer_AI\y-hyun\testvideo\test6.mp4")
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    
    results = model(frame)

    heatmap = np.zeros_like(frame[:, :, 0], dtype=np.float32)

    for result in results:
        for obj in result:
            if obj["label"] == "person":
                x1, y1, x2, y2 = obj["box"]
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                heatmap[int(y1):int(y2), int(x1):int(x2)] += 1

    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_JET)

    result_frame = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

    video_writer.write(result_frame)

    cv2.imshow("Frame", result_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()

cv2.destroyAllWindows()
