import json
import os
import cv2


def load_json(json_file):
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            contour = json.load(f)
    else:
        print("{} is not exits".format(json_file))
        contour = None
    return contour


if __name__ == "__main__":
    file_name = os.path.join(r"\\192.168.20.63\ai\Liyou_wang_data\double_cameras_video\imgs\xy1",
                             "18284518_rot_outline_4200_4800_下眼皮.json")
    contours = load_json(file_name)
    cv2.findContours()
    print(type(contours[0]))