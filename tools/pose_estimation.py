import argparse
import glob
import os
import sys

import cv2
import numpy

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pose2img


def main(args: argparse.Namespace):
    openpose = pose2img.posemodel.OpenPose(args.proto2d, args.model2d, thr=0.01)
    for path in glob.glob(args.target):
        print(path)
        process_image(openpose, path)


def process_image(openpose, image):
    basename = os.path.splitext(image)[0]
    img = cv2.imread(image)
    predicted = openpose.predict(img)
    result = openpose.show(predicted, img)
    cv2.imwrite("{}_pose.png".format(basename), result)
    numpy.save(open("{}_pose.npz".format(basename), "wb+"), predicted)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto2d', help='Path to .prototxt', required=True)
    parser.add_argument('--model2d', help='Path to .caffemodel', required=True)
    parser.add_argument("target")
    args = parser.parse_args()
    main(args)
