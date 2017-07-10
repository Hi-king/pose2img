# -*- coding: utf-8 -*-
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pose2img
import argparse
import chainer
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--encoder", required=True)
parser.add_argument("--decoder", required=True)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("image_directory")
parser.add_argument("output_directory")
args = parser.parse_args()
os.makedirs(args.output_directory)

encoder = pose2img.models.Encoder(in_ch=3)
chainer.serializers.load_npz(args.encoder, encoder)
decoder = pose2img.models.Decoder(out_ch=3)
chainer.serializers.load_npz(args.decoder, decoder)

if args.gpu >= 0:
    chainer.cuda.check_cuda_available()
    encoder.to_gpu(args.gpu)
    decoder.to_gpu(args.gpu)
    xp = chainer.cuda.cupy
else:
    import numpy
    xp = numpy

for path in glob.glob("{}/*".format(args.image_directory)):
    variable = pose2img.utility.image2variable(path)
    if args.gpu >= 0:
        variable.to_gpu(args.gpu)

    converted = decoder(encoder(variable, test=True), test=True)
    image = pose2img.utility.variable2img(converted)
    image.save("{}/{}".format(args.output_directory, os.path.basename(path)))
