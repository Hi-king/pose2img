# -*- coding: utf-8 -*-
from PIL import Image
import numpy
import chainer


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
