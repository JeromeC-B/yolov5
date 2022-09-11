from pathlib import Path

import pandas as pd
import numpy as np
import re
import ast


"""
https://www.kaggle.com/code/jeffaudi/aircraft-detection-with-yolov5
"""


IMG_SIZE = (2560, 2560)


def format_geometry(x):
    return ast.literal_eval(x.rstrip('\r\n'))


def get_bounds(geometry):
    try:
        arr = np.array(geometry).T
        xmin = np.min(arr[0])
        ymin = np.min(arr[1])
        xmax = np.max(arr[0])
        ymax = np.max(arr[1])
        return (xmin, xmax, ymin, ymax)
    except:
        raise Exception("pas normal?")
        return np.nan


def get_label(geometry):
    xmin, xmax, ymin, ymax = get_bounds(geometry)  # (xmin, xmax, ymin, ymax)

    # ### le format de yolov5 c'est avec un y inversé....
    # https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
    ymin = IMG_SIZE[0] - ymin
    ymax = IMG_SIZE[0] - ymax

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
        # ### avec ce label, il y a un problème dans le df annotations.csv
        # if name == "1e7e0450-6eb3-479e-88c2-990abc8207fa":
        #     t = 0
        # ### avec ce label, il y a un problème dans le df annotations.csv

        label = get_label(geometry=df["geometry"].iloc[i])
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
