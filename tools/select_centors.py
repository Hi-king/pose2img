# -*- coding: utf-8 -*-
import argparse
import glob
import os
import json
import numpy
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("keypoints_directory")
parser.add_argument("source_directory")
parser.add_argument("target_directory")
args = parser.parse_args()

for keypoint_path in glob.glob("{}/*.json".format(args.keypoints_directory)):
    d = json.load(open(keypoint_path))
    peoples = d["people"]

    basename = os.path.splitext(os.path.basename(keypoint_path))[0]
    baseid = basename.split("_")[0]
    print(baseid)

    source_path = glob.glob("{}/{}*".format(args.source_directory, baseid))[0]
    target_path = os.path.join(args.target_directory, os.path.basename(source_path))

    # 画面に一人
    if not len(peoples) == 1:
        continue

    body = peoples[0]["body_parts"]
    vecs = []
    for i in range(0, len(body), 3):
        vec = body[i:i + 2]
        if vec[0] > 1 and vec[1] > 1:
            vecs.append(vec)
    vecs = numpy.array(vecs)
    xmin = vecs[:, 0].min()
    xmax = vecs[:, 0].max()
    ymin = vecs[:, 1].min()
    ymax = vecs[:, 1].max()

    source_img = cv2.imread(source_path)
    source_height, source_width = source_img.shape[:2]

    width = xmax - xmin
    height = ymax - ymin
    width_ratio = float(width)/source_width
    height_ratio = float(height)/source_height

    if width_ratio < 0.1 or height_ratio < 0.1:
        continue

    os.link(source_path, target_path)


