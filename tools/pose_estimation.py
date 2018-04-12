import argparse
import cv2
import sys

import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pose2img

def main(args: argparse.Namespace):
    openpose = pose2img.posemodel.OpenPose(args.proto2d, args.model2d, thr=0.01)
    img = cv2.imread(args.image)
    predicted = openpose.predict(img)
    print(predicted)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto2d', help='Path to .prototxt', required=True)
    parser.add_argument('--model2d', help='Path to .caffemodel', required=True)
    parser.add_argument("image")
    args = parser.parse_args()
    main(args)
