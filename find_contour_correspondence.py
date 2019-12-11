import json
import os
import cv2
import numpy as np


def load_json(json_file):
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            contour = json.load(f)
    else:
        print("{} is not exits".format(json_file))
        contour = None
    return contour


if __name__ == "__main__":
    """
    step1: 计算弧长
    step2: 基于关键点计算弧长
    step3: 插值出指定对应点
    step4: 基于关键点拟合轮廓线
    """
    file_name = os.path.join(r"\\192.168.20.63\ai\Liyou_wang_data\double_cameras_video\imgs\xy1",
                             "18284518_rot_outline_4200_4800_下眼皮.json")
    img_root_path = r"\\192.168.20.63\ai\Liyou_wang_data\double_cameras_video\imgs" \
                    r"\xy1\18284518_rot_outline_4200_4800"
    eye_root_path = r"\\192.168.20.63\ai\Liyou_wang_data\double_cameras_video\imgs\xy1\18286112_rot_eye\left"
    img_names = os.listdir(r"\\192.168.20.63\ai\Liyou_wang_data\double_cameras_video\imgs"
                           r"\xy1\18284518_rot_outline_4200_4800")
    contours = load_json(file_name)
    markers_contour = np.array(contours[0]['landmarks'])
    print(markers_contour[:, 0].T.shape)
    markers_contour = markers_contour[np.lexsort(markers_contour[:, 0][np.newaxis,:])]

    left_contour = markers_contour[np.where(markers_contour[:, 0] < 500)[0], :]
    right_contour = markers_contour[np.where(markers_contour[:, 0] > 500)[0], :]
    approx_curve = cv2.approxPolyDP(markers_contour, epsilon=0.8, closed=False)
    img = cv2.imread(os.path.join(img_root_path, "{}".format(img_names[0])))
    print(approx_curve.shape)
    print("markers shape is {}".format(markers_contour.shape))
    cv2.polylines(img, [left_contour, right_contour], isClosed=False, color=(0, 255, 0), thickness=1, lineType=8, shift=0)
    # cv2.drawContours(img, [left_contour, right_contour], contourIdx=-1, color=(0, 255, 0))
    # print(type(contours[0]))
    cv2.imshow("test", img)
    cv2.waitKey(0)
    for i in range(0, 1000):
        eye_img = cv2.imread(os.path.join(img_root_path, "{}.jpg".format(4500)))
        imgray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.Canny(imgray, 90, 255)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(len(contours))
        cnt = contours[6]
        print(cnt)
        # cv2.drawContours(eye_img, contours, -1, (0, 255, 0), 3)
        cv2.drawContours(eye_img, [cnt], 0, (255, 255, 0), 3)
        cv2.imshow("test", thresh)
        cv2.waitKey(0)

