import cv2
from ultralytics import YOLO
import time
import numpy as np
import cv2
from custom_socket import CustomSocket
import socket
import json
import numpy as np
import traceback
from tensorflow import keras
import tensorflow as tf
import math


WEIGHT = "weights/rbc2023-pose.pt"
KERAS_WEIGHT = "weights/rbc2023-bae-n-noconf.h5"
# DATASET_NAME = "coco"
# DATASET_NAME = {0: "coke"}
DATASET = ["red_wine", "cola", "tropical_juice", "milk", "iced_tea", "orange_juice", "tomato_soup", "strawberry_jello", "chocolate_jello", "sugar", "rubiks_cube", "pringles", "cornflakes", "cheezit", "chocolate_cornflakes"]
# DATASET_NAME = {0: "snack", 1: "juice"}
# YOLOV8_CONFIG = {"tracker": "botsort.yaml",
#                  "conf": 0.7,
#                  "iou": 0.3,
#                  "show": False,
#                  "verbose": False}

BOX_INDEX = [5, 7, 8, 9, 10, 12, 13, 14]

YOLO_CONF = 0.7
KEYPOINTS_CONF = 0.7

JOINT_LINES = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4),(1,7),(2,6),(3,5)]

# xyz-x-y-z
FACES = [(0, 1, 2, 3), (0, 3, 5, 4), (0, 1, 7, 4),
         (4, 5, 6, 7), (1, 2, 6, 7), (2, 3, 5, 6)]
V_FOV = 72
H_FOV = 104.5


def process_keypoints(keypoints, conf, frame_width, frame_height, origin=(0, 0)):
    kpts = np.copy(keypoints)
    kpts[:, 0] = (kpts[:, 0] - origin[0]) / frame_width
    kpts[:, 1] = (kpts[:, 1] - origin[1]) / frame_height

    kpts[:, :-1][kpts[:, 2] < conf] = [-1, -1]
    return np.round(kpts[:, :-1].flatten(), 4)

def draw_points(frame, keypoints, color=(0, 255, 255)):
    for i, pt in enumerate(keypoints):
        x, y, conf = pt
        if conf > KEYPOINTS_CONF:
            cv2.putText(frame, str(i), (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_3d_lines(frame, keypoints, joint_list):
    for joint in joint_list:
        cv2.line(frame, [int(k) for k in keypoints[joint[0]][:2]], [
                 int(k) for k in keypoints[joint[1]][:2]], (255, 255, 0), 1)


def get_face_center(keypoints, faces_index_list):
    face_centroid = []
    visible_face = []
    for i, face in enumerate(faces_index_list):
        # corner_coord = np.array([keypoints[index] for index in face])
        # # print(corner_coord)
        # # print(np.mean(corner_coord, axis=0))
        visible = np.all(keypoints[list(face), 2] > KEYPOINTS_CONF)
        if visible:
            visible_face.append(["+x","+y","+z","-x","-y","-z"][i])
        face_centroid.append(list(np.mean(keypoints[list(face), :2], axis=0)) + [int(visible)])
    return np.array(face_centroid), visible_face


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


def normalize(vectors):
    re_vectors = vectors.reshape((3, 3))
    magnitude = np.linalg.norm((re_vectors), axis=0)
    print(magnitude)
    unit_vector = re_vectors / magnitude
    return unit_vector.flatten()


def process_kpts_with_label_angle(kpts, label, angle):
    x1, y1 = np.min(kpts[:, 0]), np.min(kpts[:, 1])
    x2, y2 = np.max(kpts[:, 0]), np.max(kpts[:, 1])

    copy_kpts = np.copy(kpts[:, :2])
    copy_kpts[:, 0] = (copy_kpts[:, 0] - x1) / (x2-x1)
    copy_kpts[:, 1] = (copy_kpts[:, 1] - y1) / (y2-y1)
    # copy_kpts[:, 2] = kpts[:, 2] > KEYPOINTS_CONF

    return np.concatenate(([label], angle,  copy_kpts.flatten())).reshape((1,19))

def get_rel_angle(center, v_fov, h_fov):
    cx, cy = center
    x_angle = math.atan((2*cx - 1) * math.tan(h_fov / 2 * math.pi / 180)) * 180 / math.pi
    y_angle = math.atan((2*cy - 1) * math.tan(v_fov / 2 * math.pi / 180)) * 180 / math.pi
    return (x_angle, y_angle)


def main():
    HOST = socket.gethostname()
    PORT = 12305

    server = CustomSocket(HOST, PORT)
    server.startServer()

    print("Loading YOLO")
    model = YOLO(WEIGHT, task="pose")
    print("DONE")

    # Limit Keras GPU Usage
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs, ",
                  len(logical_gpus), "Logical GPUs")

        except RuntimeError as e:
            print(e)

    print("Loading Keras Model")
    try:
        keras_model = keras.models.load_model(KERAS_WEIGHT, compile=False)
        pred_keras = True
        print("DONE")
    except:
        print("Error while loading keras model")
        pred_keras = False
        keras_model = ""

    while True:
        # Wait for connection from client :}
        conn, addr = server.sock.accept()
        print("Client connected from", addr)

        # start = time.time()

        # Process frame received from client
        while True:
            res = dict()
            msg = {"res": res}

            try:
                data = server.recvMsg(
                    conn, has_splitter=True, has_command=False)
                frame_height, frame_width, img = data

                msg["camera_info"] = [frame_width, frame_height]

                results = model.track(
                    source=img, conf=YOLO_CONF, show=False, verbose=False, persist=True, imgsz=(frame_width, frame_height))[0]
                kpts = results.keypoints.cpu().numpy()
                boxes = results.boxes.data.cpu().numpy()

                for obj_kpts, obj_box in zip(kpts, boxes):
                    obj_res = dict()
                    x1, y1, x2, y2 = obj_box[:4]
                    cx, cy = x1 + (x2-x1)/2, y1 + (y2-y1)/2
                    obj_id = int(obj_box[4])
                    obj_class = int(obj_box[-1])

                    obj_res["bbox"] = (int(x1), int(y1), int(x2), int(y2))
                    obj_res["center"] = (int(cx), int(cy))
                    obj_res["name"] = DATASET[obj_class]

                    if obj_class in BOX_INDEX:
                        fov_angle = get_rel_angle((cx/frame_width,cy/frame_height), V_FOV, H_FOV)
                        keras_input = process_kpts_with_label_angle(obj_kpts, obj_class, fov_angle)
                        print(keras_input)
                        pred_axis = normalize(keras_model.predict(keras_input, verbose=0))

                        reshape_axis = pred_axis.reshape((3,3))

                        obj_res["normal0"] = [float(i) for i in reshape_axis[0]]
                        obj_res["normal1"] = [float(i) for i in reshape_axis[1]]
                        obj_res["normal2"] = [float(i) for i in reshape_axis[2]]
                        obj_res["frame_dim"] = (frame_height, frame_width)

                        # print(fov_angle)
                        # print(reshape_axis)
                        draw_axis(img, pred_axis, center=(int(cx), int(cy)))
                        draw_3d_lines(img, obj_kpts, JOINT_LINES)

                        face_center, visible_face = get_face_center(obj_kpts, FACES)

                        obj_res["face_center"] = face_center.astype("uint8").tolist()
                        obj_res["visible_face"] = visible_face

                        draw_points(img, face_center, color = (125, 0, 255))


                    # Draw
                    cv2.rectangle(img, (int(x1), int(y1)),
                                (int(x2), int(y2)), (255, 0, 255), 1)
                    draw_points(img, obj_kpts)

                    res[obj_id] = obj_res

                # print(res)
                msg['res'] = res

                cv2.imshow("frame", img)
                cv2.waitKey(1)

                # Send back result
                # print(res)
                server.sendMsg(conn, json.dumps(msg))

            except Exception as e:
                traceback.print_exc()
                print(e)
                print("Connection Closed")
                del res
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
