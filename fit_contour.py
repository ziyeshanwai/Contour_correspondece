import os
import cv2
import json
import numpy as np
from scipy import interpolate as intp
from scipy.interpolate import UnivariateSpline


def load_json(json_file):
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            json_data = json.load(f)
    else:
        print("{} is not exits".format(json_file))
        json_data = None
    return json_data


if __name__ == "__main__":
    kv_root_path = r'\\192.168.20.63\ai\Liyou_wang_data\double_cameras_video\imgs\xy1\18286112_rot_optical_flow'
    fit_number = 20
    poly_num = 2  # 多项式最高次数
    for i in range(0, 1000):
        kv_data = load_json(os.path.join(kv_root_path, "{}_smooth.kv".format(i)))
        contour_list = []
        for j in [k for k in range(187, 195)]:
            contour_list.append(kv_data[str(j)])

        # cv2.drawContours(img, [left_contour, right_contour], contourIdx=-1, color=(0, 255, 0))
        img = cv2.imread(os.path.join(kv_root_path, "{}.jpg".format(i)))
        contours_array = np.array(contour_list)
        print(contours_array.shape)
        """
        method 1 use np.polyfit fit 5 points
        """
        # approx_curve_func = np.polyfit(contours_array[:, 0], contours_array[:, 1], poly_num)  # fitting
        # contours_fit = np.zeros((fit_number, 2), dtype=np.float32)
        # contours_fit[:, 0] = np.linspace(contours_array[0, 0], contours_array[-1, 0], num=fit_number)  # x
        #
        # contours_fit[:, 1] = np.polyval(approx_curve_func, np.linspace(contours_array[0, 0],
        #                                                                contours_array[-1, 0], num=fit_number))  # y
        # contours_fit = np.rint(contours_fit).astype(np.int32)  # 四舍五入取整

        """
        method2 use numpy.polynomial to fit 
        """
        # approx_curve_func = intp.interp1d(contours_array[:, 0], contours_array[:, 1], kind='quadratic')
        # contours_fit = np.zeros((fit_number, 2), dtype=np.float32)
        # contours_fit[:, 0] = np.linspace(contours_array[0, 0], contours_array[-1, 0], num=fit_number)  # x
        #
        # contours_fit[:, 1] = approx_curve_func(np.linspace(contours_array[0, 0],
        #                                                    contours_array[-1, 0], num=fit_number))  # y
        # contours_fit = np.rint(contours_fit).astype(np.int32)  # 四舍五入取整

        """
        method3 use scipy.interpolate import UnivariateSpline to smooth
        """
        # approx_curve_func = UnivariateSpline(contours_array[:, 0], contours_array[:, 1])
        # approx_curve_func.set_smoothing_factor(1.5)
        # contours_fit = np.zeros((fit_number, 2), dtype=np.float32)
        # contours_fit[:, 0] = np.linspace(contours_array[0, 0], contours_array[-1, 0], num=fit_number)  # x
        #
        # contours_fit[:, 1] = approx_curve_func(np.linspace(contours_array[0, 0],
        #                                                    contours_array[-1, 0], num=fit_number))  # y
        # contours_fit = np.rint(contours_fit).astype(np.int32)  # 四舍五入取整

        """
        method4 use ellipse to fit
        """
        ellipse = cv2.fitEllipse(contours_array)
        cv2.ellipse(img, ellipse, (0, 255, 0), 2)

        # cv2.polylines(img, [contours_array], isClosed=False, color=(0, 0, 255), thickness=1, lineType=8,
        #               shift=0)
        # cv2.polylines(img, [contours_fit], isClosed=False, color=(0, 255, 0), thickness=1, lineType=8,
        #               shift=0)
        cv2.imshow("test", img)
        cv2.waitKey(0)