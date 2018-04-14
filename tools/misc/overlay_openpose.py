import argparse
import cv2


def main(args):
    original = cv2.imread(args.original)
    openpose = cv2.imread(args.openpose)

    h, w = original.shape[:2]
    rh, rw = openpose.shape[:2]

    openpose = cv2.resize(openpose, (int(rw * float(h) / rh), h))[:, :w, :]
    print(original.shape)
    print(openpose.shape)
    original[openpose > 0] = openpose[openpose > 0]

    cv2.imwrite("result.png", original)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("original")
    parser.add_argument("openpose")
    args = parser.parse_args()
    main(args)
