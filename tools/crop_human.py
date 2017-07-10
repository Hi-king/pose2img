import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pose2img
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("json")
parser.add_argument("img")
parser.add_argument("out_img")
args = parser.parse_args()

pose2img.utility.crop_human(args.json, args.img, args.out_img)
