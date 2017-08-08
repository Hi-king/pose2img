# -*- coding: utf-8 -*-
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pose2img
import argparse
import chainer
import numpy

parser = argparse.ArgumentParser()
parser.add_argument("--encoder", required=True)
parser.add_argument("--decoder", required=True)
parser.add_argument("--outfile", default="output.png")
parser.add_argument("--noise_dimention", required=True, type=int)
parser.add_argument("image")
args = parser.parse_args()

encoder = pose2img.models.Encoder(in_ch=3)
chainer.serializers.load_npz(args.encoder, encoder)

decoder = pose2img.models.NoiseDecoder(noise_dimention=args.noise_dimention, out_ch=3)
chainer.serializers.load_npz(args.decoder, decoder)

variable = pose2img.utility.image2variable(args.image)

IMAGE_SHAPE = (4, 4)
images = []
with chainer.using_config('train', False):
    encoded = encoder(variable)
    encoded_batch = [
        chainer.functions.stack([encoded_each] * numpy.product(IMAGE_SHAPE), axis=1)[0]
        for encoded_each in encoded
    ]
    noise = chainer.functions.stack(
        [decoder.create_noise(encoded[-1].shape) for _ in range(numpy.product(IMAGE_SHAPE))], axis=1)[0]
    converted = decoder(encoded_batch, noise=noise)
    images = [
        pose2img.utility.array2img(converted.data[i])
        for i in range(numpy.product(IMAGE_SHAPE))
        ]
pose2img.utility.concat_images(images, IMAGE_SHAPE).save(args.outfile)
