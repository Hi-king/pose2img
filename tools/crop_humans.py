# -*- coding: utf-8 -*-
import glob
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pose2img
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("keypoints_directory")
parser.add_argument("source_directory")
parser.add_argument("target_directory")
args = parser.parse_args()

for keypoint_path in glob.glob("{}/*.json".format(args.keypoints_directory)):
    basename = os.path.splitext(os.path.basename(keypoint_path))[0]
    baseid = basename.split("_")[0]
    print(baseid)

    source_path = "{}/{}.jpg".format(args.source_directory, baseid)
    target_path = "{}/{}.png".format(args.target_directory, baseid)
    pose2img.utility.crop_human(keypoint_path, source_path, target_path)
