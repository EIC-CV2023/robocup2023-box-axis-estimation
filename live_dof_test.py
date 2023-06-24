import cv2
import numpy as np
import os

FILE_ID = 0

def draw_3d(frame, keypoints, joint_list):
    for joint in joint_list:
        cv2.line(frame, keypoints[joint[0]], keypoints[joint[1]], (255,0,0),1)

def dof(frame, vector_dir, max_len=50, color=(255,0,0), center="default"):
    x, y, z = vector_dir
    if center == "default":
        center = int(frame.shape[0] // 2)
        cv2.line(frame, (center, center), (int(center + y * max_len), int(center - z * max_len)), color, 2)
    else:
        cv2.line(frame, center, (int(center[0] + y * max_len), int(center[1] - z * max_len)), color, 2)



def draw_dof(frame, vector_out):
    v_out = np.reshape(vector_out,(6,3))
    print(v_out)
    dof(frame, v_out[0], color=(0,0,255))
    dof(frame, v_out[1], color=(0,255,0))
    dof(frame, v_out[2], color=(255,0,0))
    dof(frame, v_out[3], color=(0,0,128))
    dof(frame, v_out[4], color=(0,128,0))
    dof(frame, v_out[5], color=(128,0,0))

def normalize(vectors):
  re_vectors = vectors.reshape((3,3))
  magnitude = np.linalg.norm((re_vectors), axis=0)
  print(magnitude)
  unit_vector = re_vectors / magnitude
  return unit_vector.flatten()

while True:
    FILE = str(FILE_ID).zfill(6)
    IMAGE_PATH = os.path.join("output", "ojo_pose2", f"{FILE}.rgb.png")
    DOF_LABEL_PATH = os.path.join("output", "ojo_pose2", f"dof_{FILE}.rgb.txt")
    LABEL_PATH = os.path.join("output", "ojo_pose2", f"{FILE}.rgb.txt")

    img = cv2.imread(IMAGE_PATH)
    img_sz = img.shape[0]
    print(img_sz)

 
    with open(LABEL_PATH, "r") as f:
        data = f.readline().strip().split(" ")
        # print(data)
        f.close()

    cx, cy, w, h = map(float,data[1:5])
    x1, y1 = int((cx - (w/2))*img_sz), int((cy - (h/2))*img_sz)
    x2, y2 = int(x1 + w * img_sz), int(y1 + h * img_sz)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

    kpts = np.array(list(map(lambda x: int(float(x) * img_sz),data[5:]))).reshape((8,2))
    print(kpts)

    for i, kpt in enumerate(kpts):
        cv2.putText(img, str(i), kpt, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)

    draw_3d(img, kpts, [(0,1),(2,3),(4,5),(6,7),(1,2),(0,3),(5,6),(4,7),(0,7),(1,6),(2,5),(3,4)])

    with open(DOF_LABEL_PATH, "r") as f:
        dof_data = eval(f.readline().strip())
        # print(data)
        f.close()
    draw_dof(img, dof_data)

    cv2.imshow("img", img)
    cv2.waitKey()
    FILE_ID += 1
cv2.destroyAllWindows()
