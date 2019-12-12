import os
import cv2
import json
import numpy as np


if __name__ == "__main__":
    """
    根据关键点进行椭圆拟合
    """
    fit_number = 20
    poly_num = 4

    img = cv2.imread(os.path.join("./img", "{}.png".format("eye_1")))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("origin image", img_gray)
    # thresh = cv2.Canny(img_gray, 120, 180)
    _, thresh = cv2.threshold(img_gray, 30, 200, cv2.THRESH_BINARY)
    kernel_0 = np.ones((4, 4), np.uint8)
    # thresh = cv2.erode(thresh, kernel)
    thresh = cv2.dilate(thresh, kernel_0)
    kernel_1 = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel_1)
    thresh = cv2.erode(thresh, kernel_1)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    #
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Morphological opening: Get rid of the stuff at the top of the circle
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12)))

    cv2.imshow("canny image", thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    for i in range(0, len(contours)):
        cnt = contours[0]

    # cv2.drawContours(eye_img, contours, -1, (0, 255, 0), 3)
    # cv2.polylines(img, [cnt], isClosed=False, color=(0, 0, 255), thickness=1, lineType=8,
    #               shift=0)
    # cnt_array = np.array(cnt).reshape(-1, 2)
    # print(cnt_array)
    # approx_curve_func = np.polyfit(cnt_array[:, 0], cnt_array[:, 1], poly_num)  # fitting
    # contours_fit = np.zeros((fit_number, 2), dtype=np.float32)
    # contours_fit[:, 0] = np.linspace(cnt_array[0, 0], cnt_array[-1, 0], num=fit_number)  # x
    #
    # contours_fit[:, 1] = np.polyval(approx_curve_func, np.linspace(cnt_array[0, 0],
    #                                                                cnt_array[-1, 0], num=fit_number))  # y
    # contours = np.rint(contours_fit).astype(np.int32)  # 四舍五入取整
    """
    拟合椭圆
    """
    ellipse = cv2.fitEllipse(cnt)
    cv2.ellipse(img, ellipse, (0, 255, 0), 2)

    cv2.drawContours(img, contours, -1, (255, 255, 0), 3)
    cv2.imshow("contours", img)
    cv2.waitKey(0)