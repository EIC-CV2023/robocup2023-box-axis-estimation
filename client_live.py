import socket
import cv2
import numpy as np
import time
from custom_socket import CustomSocket
import json


def list_available_cam(max_n):
    list_cam = []
    for n in range(max_n):
        cap = cv2.VideoCapture(n)
        ret, _ = cap.read()

        if ret:
            list_cam.append(n)
        cap.release()

    if len(list_cam) == 1:
        return list_cam[0]
    else:
        print(list_cam)
        return int(input("Cam index: "))


host = socket.gethostname()
port = 12305

c = CustomSocket(host, port)
c.clientConnect()

cap = cv2.VideoCapture(list_available_cam(10))
cap.set(4, 480)
cap.set(3, 640)

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    # cv2.imshow('client_cam', frame)

    msg = c.req(frame)

    print(msg)

    if cv2.waitKey(1) == ord("q"):
        cap.release()

cv2.destroyAllWindows()
