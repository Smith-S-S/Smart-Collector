import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
import math
import time
import websocket
import imutils
import urllib.request

# Connect to WebSocket server
ws = websocket.WebSocket()
ws.connect("ws://192.168.178.132") #if the ipadress from the ipwebcam apk from playstore to acceaa the mobile camera
print("Connected to WebSocket server")

url="http://192.168.178.99:8080/shot.jpg"

tolerance_value = 5
new_frame_time = 0
prev_frame_time = 0
model = YOLO("best.pt")
xx = "C:/mac/pycham/pythonProject3/major_ma/data.mp4" #add out video for trial
frame = cv2.VideoCapture(xx) #if youe use 1 or 0 it use ur web came instead of (xx)
pixels_per_cm = 10

classNames = ["bottle", "plastic","paper"]

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

while True:
    imgPath = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)

    img = cv2.imdecode(imgNp, -1)
    cam = imutils.resize(img, width=1080)

    result = model.predict(source=cam, imgsz=[184, 280], rect=True,conf=0.5)
    frame_center = (cam.shape[1] // 2, cam.shape[0] // 2)

    cv2.line(cam, (0, frame_center[1]), (cam.shape[1], frame_center[1]), (0, 0, 255), 2)  # x-axis (red)
    cv2.line(cam, (frame_center[0], 0), (frame_center[0], cam.shape[0]), (0, 0, 255), 2)

    nearest_distance = float('inf')
    nearest_quadrant = ""
    nearest_center = None

    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)
                # y                                      x
            if x_center < frame_center[0] and y_center < frame_center[1]:
                quadrant = "1st"
            elif x_center >= frame_center[0] and y_center < frame_center[1]:
                quadrant = "2nd"
            elif x_center < frame_center[0] and y_center >= frame_center[1]:
                quadrant = "3rd"
            else:
                quadrant = "4th"

            distance = calculate_distance(frame_center, (x_center, y_center))
            distance_cm = round(distance / pixels_per_cm)

            if distance_cm < nearest_distance:
                nearest_distance = distance_cm
                nearest_quadrant = quadrant
                nearest_center = (x_center, y_center)

            cv2.rectangle(cam, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 5)

            cvzone.putTextRect(cam, f'Waste: {classNames[int(box.cls)]}, Quadrant: {quadrant}, Distance: {distance:.2f} cm',
                               (max(0, int(x1)), max(35, int(y1))), scale=2, thickness=3, offset=10)

    if nearest_center:
        cv2.circle(cam, nearest_center, 10, (0, 0, 255), -1)

        deviation_x = nearest_center[0] - frame_center[0]
        deviation_y = nearest_center[1] - frame_center[1]

        if abs(deviation_x) > tolerance_value:
            movement_direction = "LEFT" if deviation_x > 0 else "RIGHT"
        else:
            movement_direction = "Forward" if deviation_y > 0 else "Forward"

        print(f"Deviation : {movement_direction}")
        cv2.putText(cam, f'Direction To Move: {movement_direction}', (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0),
                    2)
        ws.send(movement_direction)

    else:

        movement_direction = "Forward"
        print(f"Deviation : {movement_direction}")
        cv2.putText(cam, f'Direction To Move: {movement_direction}', (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0),
                    2)
        # Send movement direction to ESP32 via WebSocket
        ws.send(movement_direction)

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print("FPS: ", int(fps))
    cv2.imshow("Debris Collector", cam)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Gracefully close WebSocket connection
ws.close()
cv2.destroyAllWindows()
