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
from tensorflow import keras
import tensorflow as tf
import math

JOINT_LINES = [(0, 1), (2, 3), (4, 5), (6, 7), (1, 2),
               (0, 3), (5, 6), (4, 7), (0, 7), (1, 6), (2, 5), (3, 4)]
FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 3, 4, 7),
         (1, 2, 5, 6), (0, 1, 6, 7), (2, 3, 4, 5)]

V_FOV = 72
H_FOV = 104.5

gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs, ", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e:
        print(e)


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


def draw_points(frame, keypoints, color=(0, 255, 255)):
    for i, pt in enumerate(keypoints):
        x, y = pt
        cv2.putText(frame, str(i), (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_3d_lines(frame, keypoints, joint_list):
    for joint in joint_list:
        cv2.line(frame, [int(k) for k in keypoints[joint[0]]], [
                 int(k) for k in keypoints[joint[1]]], (255, 255, 0), 1)


def get_face_center(keypoints, faces_index_list):
    face_centroid = []
    for face in faces_index_list:
        corner_coord = np.array([keypoints[index] for index in face])
        # print(corner_coord)
        # print(np.mean(corner_coord, axis=0))
        face_centroid.append(np.mean(corner_coord, axis=0))
    return np.array(face_centroid)


def axis(frame, vector_dir, max_len=75, color=(255, 0, 0), center="default"):
    x, y, z = vector_dir
    if center == "default":
        center = int(frame.shape[0] // 2)
        cv2.line(frame, (center, center), (int(center + y * max_len),
                 int(center - z * max_len)), color, 3)
    else:
        cv2.line(frame, center, (int(
            center[0] + y * max_len), int(center[1] - z * max_len)), color, 3)


def draw_axis(frame, vector_out, center="default"):
    v_out = np.reshape(vector_out, (3, 3))
    print(v_out)
    axis(frame, v_out[0], color=(0, 0, 255), center=center)
    axis(frame, v_out[1], color=(0, 255, 0), center=center)
    axis(frame, v_out[2], color=(255, 0, 0), center=center)
    # dof(frame, v_out[3], color=(0,0,128), center=center)
    # dof(frame, v_out[4], color=(0,128,0), center=center)
    # dof(frame, v_out[5], color=(128,0,0), center=center)


def normalize(vectors):
    re_vectors = vectors.reshape((3, 3))
    magnitude = np.linalg.norm((re_vectors), axis=0)
    print(magnitude)
    unit_vector = re_vectors / magnitude
    return unit_vector.flatten()


# def process_kpts(kpts):
#     x1, y1 = np.min(kpts[:, 0]), np.min(kpts[:, 1])
#     x2, y2 = np.max(kpts[:, 0]), np.max(kpts[:, 1])

#     copy_kpts = np.copy(kpts)
#     copy_kpts[:, 0] = (copy_kpts[:, 0] - x1) / (x2-x1)
#     copy_kpts[:, 1] = (copy_kpts[:, 1] - y1) / (y2-y1)

#     return copy_kpts.reshape((1,16))

# def process_kpts_with_label(kpts, label):
#     x1, y1 = np.min(kpts[:, 0]), np.min(kpts[:, 1])
#     x2, y2 = np.max(kpts[:, 0]), np.max(kpts[:, 1])

#     copy_kpts = np.copy(kpts)
#     copy_kpts[:, 0] = (copy_kpts[:, 0] - x1) / (x2-x1)
#     copy_kpts[:, 1] = (copy_kpts[:, 1] - y1) / (y2-y1)

#     return np.concatenate(([label], copy_kpts.flatten())).reshape((1,17))

def process_kpts_with_label_angle(kpts, label, angle):
    x1, y1 = np.min(kpts[:, 0]), np.min(kpts[:, 1])
    x2, y2 = np.max(kpts[:, 0]), np.max(kpts[:, 1])

    copy_kpts = np.copy(kpts)
    copy_kpts[:, 0] = (copy_kpts[:, 0] - x1) / (x2-x1)
    copy_kpts[:, 1] = (copy_kpts[:, 1] - y1) / (y2-y1)

    return np.concatenate(([label], angle,  copy_kpts.flatten())).reshape((1,19))

def get_rel_angle(center, v_fov, h_fov):
    cx, cy = center
    x_angle = math.atan((2*cx - 1) * math.tan(h_fov / 2 * math.pi / 180)) * 180 / math.pi
    y_angle = math.atan((2*cy - 1) * math.tan(v_fov / 2 * math.pi / 180)) * 180 / math.pi
    return (x_angle, y_angle)


model = YOLO("weights/fov-oj-pose.pt", task="pose")
# cap = cv2.VideoCapture(list_available_cam(5))
cap = cv2.VideoCapture("data/dof_test_2.mp4")

FRAME_WIDTH, FRAME_HEIGHT = int(cap.get(3)), int(cap.get(4))
vid_writer = cv2.VideoWriter('pose_test.mp4', cv2.VideoWriter_fourcc(*"MJPG"), 10, (FRAME_WIDTH, FRAME_HEIGHT))

keras_model = keras.models.load_model(
    "weights/pose_fov.h5", compile=False)

YOLO_CONF = 0.7
KEYPOINTS_CONF = 0.7

start = time.time()
while cap.isOpened():
    res = dict()
    ret, frame = cap.read()
    if not ret:
        print("Error")
        continue

    # FRAME_HEIGHT, FRAME_WIDTH = frame.shape[:-1]

    # FOR ZED
    # frame = frame[:, :int(FRAME_WIDTH / 2)]
    # FRAME_WIDTH /= 2

    results = model.track(source=frame, conf=YOLO_CONF,
                          show=False, verbose=False, persist=True)[0]
    kpts = results.keypoints.cpu().numpy()
    boxes = results.boxes.data.cpu().numpy()
    # print(boxes)
    # print(kpts)
    

    for obj_kpts, obj_box in zip(kpts, boxes):
        obj_res = dict()
        x1, y1, x2, y2 = obj_box[:4]
        cx, cy = x1 + (x2-x1)/2, y1 + (y2-y1)/2
        obj_id = int(obj_box[4])
        obj_class = int(obj_box[-1])

        obj_res["bbox"] = (int(x1), int(y1), int(x2), int(y2))
        obj_res["center"] = (int(cx), int(cy))
        obj_res["class"] = obj_class



        fov_angle = get_rel_angle((cx/FRAME_WIDTH,cy/FRAME_HEIGHT), V_FOV, H_FOV)
        keras_input = process_kpts_with_label_angle(obj_kpts, 0, fov_angle)
        pred_axis = normalize(keras_model.predict(keras_input, verbose=0))

        reshape_axis = pred_axis.reshape((3,3))

        obj_res["normal0"] = reshape_axis[0]
        obj_res["normal1"] = reshape_axis[1]
        obj_res["normal2"] = reshape_axis[2]
        obj_res["frame_dim"] = (FRAME_HEIGHT, FRAME_WIDTH)

        print(fov_angle)
        print(reshape_axis)

        # Draw
        cv2.rectangle(frame, (int(x1), int(y1)),
                      (int(x2), int(y2)), (255, 0, 255), 1)
        draw_3d_lines(frame, obj_kpts, JOINT_LINES)
        draw_points(frame, obj_kpts)
        draw_axis(frame, pred_axis, center=(int(cx), int(cy)))

        res[obj_id] = obj_res

    cv2.putText(frame, "fps: " + str(round(1 / (time.time() - start), 2)), (10, int(cap.get(4)) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    print(res)

    start = time.time()

    cv2.imshow("frame", frame)

    vid_writer.write(frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        cap.release()

vid_writer.release()

cv2.destroyAllWindows()
