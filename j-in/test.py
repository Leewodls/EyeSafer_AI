import torch
import cv2
import numpy as np


model_path = r'C:\Users\dltls\EyeSafer\j-in\best.pt'  
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

alert_threshold = 3

video_path = r'C:\Users\dltls\EyeSafer\s-chul\Crowd size at Donald Trump rally in Wildwood NJ.mp4'  

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

def calculate_distance(box1, box2):
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return distance

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)

    persons = [x for x in results.xyxy[0] if int(x[-1]) == 0] 
    num_persons = len(persons)

    cv2.putText(frame, f'Total Persons: {num_persons}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    close_persons = []
    for i in range(num_persons):
        for j in range(i + 1, num_persons):
            box1 = persons[i][:4]
            box2 = persons[j][:4]
            distance = calculate_distance(box1, box2)
            if distance < 100:  
                close_persons.append((box1, box2))

    if len(close_persons) >= alert_threshold:
        centers = []
        for box1, box2 in close_persons:
            centers.append(get_center(box1))
            centers.append(get_center(box2))

        centers = np.array(centers)
        avg_center = np.mean(centers, axis=0)

        x_min = int(np.min(centers[:, 0]) - 50)
        y_min = int(np.min(centers[:, 1]) - 50)
        x_max = int(np.max(centers[:, 0]) + 50)
        y_max = int(np.max(centers[:, 1]) + 50)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 0, 255), -1)
        alpha = 0.4  
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    for *xyxy, conf, cls in persons:
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
        cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
