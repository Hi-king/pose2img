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
parser.add_argument("--noise_dimention", default=0, type=int)
parser.add_argument("image_directory")
parser.add_argument("output_directory")
args = parser.parse_args()
os.makedirs(args.output_directory)

encoder = pose2img.models.Encoder(in_ch=3)
chainer.serializers.load_npz(args.encoder, encoder)
if args.noise_dimention > 0:
    decoder = pose2img.models.NoiseDecoder(noise_dimention=args.noise_dimention, out_ch=3)
else:
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

noise = None
with chainer.using_config('train', False):
    for path in glob.glob("{}/*".format(args.image_directory)):
        variable = pose2img.utility.image2variable(path)
        if args.gpu >= 0:
            variable.to_gpu(args.gpu)

        encoded = encoder(variable)
        if args.noise_dimention > 0:
            if noise is None:
                print("create noise")
                noise = decoder.create_noise(encoded[-1].shape)
                if args.gpu >= 0:
                    noise.to_gpu(args.gpu)
            converted = decoder(encoded, noise=noise)
        else:
            converted = decoder(encoded)
        image = pose2img.utility.variable2img(converted)
        image.save("{}/{}".format(args.output_directory, os.path.basename(path)))
