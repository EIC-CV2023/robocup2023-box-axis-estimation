from ultralytics import YOLO
import cv2
import time
import numpy as np
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.torch_utils import select_device
import yaml
from random import randint
import cv2
import numpy as np
import os


JOINT_LINES = [(0, 1), (2, 3), (4, 5), (6, 7), (1, 2),
               (0, 3), (5, 6), (4, 7), (0, 7), (1, 6), (2, 5), (3, 4)]
FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 3, 4, 7),
         (1, 2, 5, 6), (0, 1, 6, 7), (2, 3, 4, 5)]


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


def draw_points(frame, keypoints, color=(0, 0, 255)):
    for i, pt in enumerate(keypoints):
        x, y = pt
        cv2.putText(frame, str(i), (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_3d_lines(frame, keypoints, joint_list):
    for joint in joint_list:
        cv2.line(frame, [int(k) for k in keypoints[joint[0]]], [
                 int(k) for k in keypoints[joint[1]]], (255, 0, 0), 2)


def get_face_center(keypoints, faces_index_list):
    face_centroid = []
    for face in faces_index_list:
        corner_coord = np.array([keypoints[index] for index in face])
        # print(corner_coord)
        # print(np.mean(corner_coord, axis=0))
        face_centroid.append(np.mean(corner_coord, axis=0))
    return np.array(face_centroid)


model = YOLO("weights/snack-pose.pt", task="pose")
# cap = cv2.VideoCapture(list_available_cam(5))
cap = cv2.VideoCapture("data/test.mov")


YOLO_CONF = 0.7
KEYPOINTS_CONF = 0.7

start = time.time()
FRAME_WIDTH = cap.get(3)
FRAME_HEIGHT = cap.get(4)

rand_color_list = np.random.rand(20, 3) * 255

while cap.isOpened():
    res = []
    ret, frame = cap.read()
    if not ret:
        print("Error")
        continue

    results = model.track(source=frame, conf=YOLO_CONF,
                          show=False, verbose=False, persist=True)[0]
    kpts = results.keypoints.cpu().numpy()
    boxes = results.boxes.data.cpu().numpy()
    # print(boxes)
    # print(kpts)

    for obj_kpts, obj_box in zip(kpts, boxes):
        # print(obj_box)
        x1, y1, x2, y2 = obj_box[:4]
        obj_id = int(obj_box[4])
        print(FRAME_WIDTH, FRAME_HEIGHT)
        # print(obj_id)

        print(obj_kpts.flatten())

        faces_centroid = get_face_center(obj_kpts, FACES)
        # print(faces_centroid)
        draw_points(frame, faces_centroid, (128, 128, 0))

        cv2.rectangle(frame, (int(x1), int(y1)),
                      (int(x2), int(y2)), (0, 255, 0), 2)

        draw_3d_lines(frame, obj_kpts, JOINT_LINES)
        draw_points(frame, obj_kpts)

    cv2.putText(frame, "fps: " + str(round(1 / (time.time() - start), 2)), (10, int(cap.get(4)) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    start = time.time()

    cv2.imshow("frame", frame)

    key = cv2.waitKey(50)

    if key == ord("q"):
        cap.release()
    if key == ord("s"):
        cv2.imwrite("frame1.jpg", frame)
        cap.release()

cv2.destroyAllWindows()
