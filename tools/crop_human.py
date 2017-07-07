import cv2
import json
import argparse
import numpy

parser = argparse.ArgumentParser()
parser.add_argument("json")
parser.add_argument("img")
parser.add_argument("out_img")
args = parser.parse_args()

raw_img = cv2.imread(args.img)

img = cv2.imread(args.img)
d = json.load(open(args.json))
print(d)
body = d["people"][0]["body_parts"]

vecs = []
for i in range(0, len(body), 3):
    vecs.append(body[i:i+2])
print(vecs)


mask = numpy.zeros(raw_img.shape)
for vec in vecs:
    if vec[0] > 1 and vec[1] > 1:
        cv2.circle(mask, tuple(map(int, vec)), radius=100, color=(255, 255, 255), thickness=-1)

raw_img[mask < 254] = 0
#cv2.imshow("test", raw_img)
#cv2.waitKey(-1)
cv2.imwrite(args.out_img, raw_img)
