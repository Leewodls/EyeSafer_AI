import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import torch
import cv2
import numpy as np
from flask import Flask, render_template, Response
import os

app = Flask(__name__)

model_path = r'C:\Users\joyon\EyeSafer_AI\j-in\best.pt'  
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

video_path = r"C:\Users\joyon\EyeSafer_AI\testvideo\test_seoul.mp4"

alert_threshold = 3  
confidence_threshold = 0.3  
distance_threshold = 150  

def calculate_distance(box1, box2):
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return distance

def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        persons = [x for x in results.xyxy[0] if int(x[-1]) == 0 and x[4] >= confidence_threshold] 
        num_persons = len(persons)

        cv2.putText(frame, f'Total Persons: {num_persons}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        clusters = []
        for i in range(num_persons):
            for j in range(i + 1, num_persons):
                box1 = persons[i][:4]
                box2 = persons[j][:4]
                distance = calculate_distance(box1, box2)
                if distance < distance_threshold:  
                    added_to_cluster = False
                    for cluster in clusters:
                        if persons[i] in cluster or persons[j] in cluster:
                            cluster.add(persons[i])
                            cluster.add(persons[j])
                            added_to_cluster = True
                            break
                    if not added_to_cluster:
                        clusters.append({persons[i], persons[j]})

        for cluster in clusters:
            if len(cluster) >= alert_threshold:
                centers = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in cluster]
                centers = np.array(centers)
                x_min = int(np.min(centers[:, 0]) - 20)
                y_min = int(np.min(centers[:, 1]) - 20)
                x_max = int(np.max(centers[:, 0]) + 20)
                y_max = int(np.max(centers[:, 1]) + 20)

                overlay = frame.copy()
                cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 0, 255), -1)
                alpha = 0.4  
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                cv2.putText(frame, 'DANGER: Crowding Detected!', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        for *xyxy, conf, cls in persons:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2])), (int(xyxy[3])), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5002)
