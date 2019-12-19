import json
import os
import cv2
import numpy as np
import pickle


def load_pickle(pickle_file):
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            mesh_contour = pickle.load(f)
    else:
        mesh_contour = None
    return mesh_contour


def load_json(json_file):
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            contour = json.load(f)
    else:
        print("{} is not exits".format(json_file))
        contour = None
    return contour


def build_correspondence(source_points, target_points):
    """
    建立对应关系，返回与source_points对应的target_points上的坐标点
    :param source_points: numpy array [n, 2] 要求排序过的
    :param target_points: numpy array [m, 2] m > n 必须的
    target_points的第一个点一定与source_points的第一个点对应
    :return: source_points corresponding_points
    """
    tangent_lines = np.diff(source_points, axis=0)
    print(tangent_lines.shape)
    source_tmp = source_points + target_points[0, :]-source_points[0, :]
    corresponding_points = list()
    corresponding_points.append(target_points[0, :])
    for i in range(1, source_tmp.shape[0]):
        vec = target_points - source_tmp[i, :]
        dis = np.abs(np.dot(vec, tangent_lines[i-1, :]))/np.linalg.norm(tangent_lines[i-1, :])
        corresponding_points.append(target_points[np.argmin(dis), :])
    return source_points, np.array(corresponding_points, dtype=np.int32)


def play_contour():
    pass



if __name__ == "__main__":
    """
    step1: 计算弧长
    step2: 基于关键点计算弧长
    step3: 插值出指定对应点
    step4: 基于关键点拟合轮廓线
    step5: 认为曲率方向与对应轮廓线的焦点即为对应点
    """
    file_name = os.path.join(r"\\192.168.20.63\ai\Liyou_wang_data\double_cameras_video\imgs\xy1",
                             "18284518_rot_outline_4200_4800_下嘴皮.json")
    img_root_path = r"\\192.168.20.63\ai\Liyou_wang_data\double_cameras_video\imgs" \
                    r"\xy1\18284518_rot_outline_4200_4800"
    eye_root_path = r"\\192.168.20.63\ai\Liyou_wang_data\double_cameras_video\imgs\xy1\18286112_rot_eye\left"
    img_names = os.listdir(r"\\192.168.20.63\ai\Liyou_wang_data\double_cameras_video\imgs"
                           r"\xy1\18284518_rot_outline_4200_4800")
    img_names = sorted(img_names)
    contours = load_json(file_name)
    fit_number = 500
    poly_num = 10  # 多项式最高次数
    start_jpg = 4200  # 其实jpg 名字
    for i in range(0, len(contours)):
        markers_contour = np.array(contours[i]['landmarks'])
        markers_contour = markers_contour[np.lexsort(markers_contour[:, 0][np.newaxis, :])]
        mesh_contour = load_pickle(os.path.join(r"\\192.168.20.63\ai\Liyou_wang_data"
                                                r"\double_cameras_video\imgs\xy1\mesh_contour_4200_4800",
                                                "right", "{}.pkl".format(i+start_jpg)))
        mesh_contour = mesh_contour[np.arange(0, mesh_contour.shape[0], 2), :]

        left_contour = markers_contour  # [np.where(markers_contour[:, 0] < 500)[0], :]
        # right_contour = markers_contour[np.where(markers_contour[:, 0] > 500)[0], :]
        # approx_curve = cv2.approxPolyDP(markers_contour, epsilon=0.8, closed=False)
        img = cv2.imread(os.path.join(img_root_path, "{}.jpg".format(contours[i]['index'] + start_jpg)))
        approx_curve_func = np.polyfit(left_contour[:, 0], left_contour[:, 1], poly_num)  # fitting
        left_contour_fit = np.zeros((fit_number, 2), dtype=np.float32)
        left_contour_fit[:, 0] = np.linspace(left_contour[0, 0], left_contour[-1, 0], num=fit_number)  # x
        left_contour_fit[:, 1] = np.polyval(approx_curve_func, np.linspace(left_contour[0, 0],
                                                                           left_contour[-1, 0], num=fit_number))  # y
        left_contour_fit = np.rint(left_contour_fit).astype(np.int32)  # 四舍五入取整
        arc_len = cv2.arcLength(left_contour, closed=False)
        source, correspondence = build_correspondence(mesh_contour, markers_contour)
        cv2.polylines(img, [markers_contour], isClosed=False, color=(0, 255, 0), thickness=1, lineType=8, shift=0)
        cv2.polylines(img, [mesh_contour], isClosed=False, color=(0, 0, 255), thickness=1, lineType=8, shift=0)
        for i in range(0, source.shape[0]):
            cv2.line(img, (source[i, 0], source[i, 1]), (correspondence[i, 0], correspondence[i, 1]),
                     color=(255, 0, 0), thickness=1)
        # cv2.polylines(img, [left_contour_fit], isClosed=False,
        #               color=(0, 0, 255), thickness=1, lineType=8, shift=0)
        # cv2.polylines(img, [right_contour], isClosed=False,
        #               color=(0, 255, 0), thickness=1, lineType=8, shift=0)
        # left_ellipse = cv2.fitEllipse(left_contour)
        # right_ellipse = cv2.fitEllipse(right_contour)
        # cv2.ellipse(img, left_ellipse, (0, 255, 0), 2)
        # cv2.ellipse(img, right_ellipse, (0, 0, 255), 2)
        # cv2.drawContours(img, [markers_contour], contourIdx=-1, color=(0, 255, 0))
        # print(type(contours[0]))
        print("show {} img".format(i))
        cv2.imshow("test", img)
        cv2.waitKey(0)


