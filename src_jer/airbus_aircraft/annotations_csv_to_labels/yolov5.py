from pathlib import Path

import pandas as pd
import numpy as np
import re
import ast


"""
https://www.kaggle.com/code/jeffaudi/aircraft-detection-with-yolov5
"""


IMG_SIZE = (2560, 2560)

# https://www.kaggle.com/code/vbookshelf/basics-of-yolo-v5-balloon-detection/notebook
import cv2
import matplotlib.pyplot as plt
def draw_bbox(image, xmin, ymin, xmax, ymax, text = None):
    """
    This functions draws one bounding box on an image.

    Input: Image (numpy array)
    Output: Image with the bounding box drawn in. (numpy array)

    If there are multiple bounding boxes to draw then simply
    run this function multiple times on the same image.

    Set text=None to only draw a bbox without
    any text or text background.
    E.g. set text='Balloon' to write a
    title above the bbox.

    xmin, ymin --> coords of the top left corner.
    xmax, ymax --> coords of the bottom right corner.

    """

    w = xmax - xmin
    h = ymax - ymin

    # Draw the bounding box
    # ......................

    start_point = (xmin, ymin)
    end_point = (xmax, ymax)
    bbox_color = (255, 0, 0)
    bbox_thickness = 15

    image = cv2.rectangle(image, start_point, end_point, bbox_color, bbox_thickness)

    # Draw the tbackground behind the text and the text
    # .................................................

    # Only do this if text is not None.
    if text:

        # Draw the background behind the text
        text_bground_color = (0, 0, 0)  # black
        cv2.rectangle(image, (xmin, ymin - 150), (xmin + w, ymin), text_bground_color, -1)

        # Draw the text
        text_color = (255, 255, 255)  # white
        font = cv2.FONT_HERSHEY_DUPLEX
        origin = (xmin, ymin - 30)
        fontScale = 3
        thickness = 10

        image = cv2.putText(image, text, origin, font, fontScale, text_color, thickness, cv2.LINE_AA)

    return image


def show_1_img(path_img, bbox_list):
    # set the figsize so the image is larger
    plt.figure(figsize=(8, 8))

    image = plt.imread(path_img)

    # Draw the bboxes on the image
    for coord_dict in bbox_list:

        xmin = int(coord_dict['xmin'])
        ymin = int(coord_dict['ymin'])
        xmax = int(coord_dict['xmax'])
        ymax = int(coord_dict['ymax'])

        image = draw_bbox(image, xmin, ymin, xmax, ymax, text=None)

    print(image.dtype)
    print(image.min())
    print(image.max())
    print(image.shape)

    plt.imshow(image)
    plt.axis('off')
    plt.show()


def format_geometry(x):
    return ast.literal_eval(x.rstrip('\r\n'))


def get_bounds(geometry):
    try:
        arr = np.array(geometry).T
        xmin = np.min(arr[0])
        xmax = np.max(arr[0])
        ymin = np.min(arr[1])
        ymax = np.max(arr[1])
        return (xmin, xmax, ymin, ymax)
    except:
        raise Exception("pas normal?")
        return np.nan


def get_label(geometry):
    xmin, xmax, ymin, ymax = get_bounds(geometry)  # (xmin, xmax, ymin, ymax)

    # ### le format des xmin, xmax, ymin, et ymax est déjà bon (si xmin=ymin=0, alors cest le coin en haut à gauche, et si
    # ### xmax==ymax==2560, cest le coin en bas à droite)

    width = np.abs(xmax - xmin)
    height = np.abs(ymax - ymin)

    x_center = (xmax + xmin) / 2
    y_center = (ymax + ymin) / 2

    # ### normalization
    x_center = x_center / IMG_SIZE[0]
    y_center = y_center / IMG_SIZE[1]

    width = width / IMG_SIZE[0]
    height = height / IMG_SIZE[1]

    return "0 {} {} {} {}".format(x_center, y_center, width, height)


def annotations_to_labels(path_in, path_out):
    """
    Ici, les truncated airplanes vont aller dans airplane class quand même

    :param path_in: les annotations.csv
    :param path_out: le dossier pour mettre les bounding boxes
    :return:
    """

    Path(path_out).mkdir(parents=True, exist_ok=True)

    """
    geometry: bounding box as a array of 5 tuples (x, y). Last element is the same as first element.
    """
    df = pd.read_csv(path_in + "/annotations.csv", converters={'geometry': format_geometry})

    info_bounding = {}
    for i in range(df.shape[0]):
        name = df["image_id"].iloc[i][:-4]
        # ### avec ce label, il y a un problème dans le df annotations.csv (trop de bounding box)
        if name == "1e7e0450-6eb3-479e-88c2-990abc8207fa":
            t = 0
            continue
        # ### avec ce label, il y a un problème dans le df annotations.csv (trop de bounding box)

        label = get_label(geometry=df["geometry"].iloc[i])
        # label_split = str.split(label)
        # img_name = "1e7e0450-6eb3-479e-88c2-990abc8207fa"
        # if name == img_name:
        #     t = 0
        #     # show_1_img(path_img=r"C:\projets\external\database\airbus-aircraft-detection\raw-data\archive\images\{}.jpg".format(img_name),
        #     #            bbox_list=[{"xmin": float(label_split[1])*IMG_SIZE[0], "xmax": float(label_split[2])*IMG_SIZE[0],
        #     #                        "ymin": float(label_split[3])*IMG_SIZE[1], "ymax": float(label_split[4])*IMG_SIZE[1]}])
        #     show_1_img(path_img=r"C:\projets\external\database\airbus-aircraft-detection\raw-data\archive\images\{}.jpg".format(img_name), bbox_list=[
        #         {"xmin": 1598, "xmax": 1689, "ymin": 873,
        #          "ymax": 938}])
        #     t = 0
        if name in info_bounding:
            if label in info_bounding[name]:
                t = 0
            info_bounding[name].append(label)
        else:
            info_bounding[name] = [label]

    for name in info_bounding:
        with open(path_out + "/{}.txt".format(name), "w") as f:
            if len(info_bounding[name]) > 30:
                t = 0
            for i in range(len(info_bounding[name])):
                if i != len(info_bounding[name]) - 1:
                    f.write(info_bounding[name][i])
                    f.write("\n")
                else:
                    f.write(info_bounding[name][i])

        t = 0
    t = 0
