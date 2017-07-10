# -*- coding: utf-8 -*-
from PIL import Image
import numpy
import chainer
import cv2
import json


def image2variable(path, resize=(256, 256)):
    image = Image.open(path).resize(resize)
    image_data = numpy.asarray(image, dtype=numpy.float32).transpose(2, 0, 1)
    if image_data.shape[0] == 4:  # RGBA
        image_data = image_data[:3]
    return chainer.Variable(numpy.array([(image_data - 128.0) / 128.0]))


def clip_img(x):
    return numpy.float32(-1 if x < -1 else (1 if x > 1 else x))


def variable2img(x) -> Image.Image:
    data = (numpy.vectorize(clip_img)(chainer.cuda.to_cpu(x.data)[0, :, :, :])).transpose(1, 2, 0)
    img = Image.fromarray(numpy.uint8((data + 1) * 128))
    return img


def crop_human(json_path, img_path, output_path):
    raw_img = cv2.imread(img_path)

    d = json.load(open(json_path))
    print(d)
    if len(d["people"]) == 0:  # no human
        return
    body = d["people"][0]["body_parts"]

    vecs = []
    for i in range(0, len(body), 3):
        vecs.append(body[i:i + 2])
    print(vecs)

    mask = numpy.zeros(raw_img.shape)
    for vec in vecs:
        if vec[0] > 1 and vec[1] > 1:
            cv2.circle(mask, tuple(map(int, vec)), radius=100, color=(255, 255, 255), thickness=-1)

    raw_img[mask < 254] = 0
    cv2.imwrite(output_path, raw_img)
