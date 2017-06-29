# -*- coding: utf-8 -*-
import pose2img
import argparse
import chainer

parser = argparse.ArgumentParser()
parser.add_argument("--encoder", required=True)
parser.add_argument("--decoder", required=True)
parser.add_argument("--outfile", default="output.png")
parser.add_argument("image")
args = parser.parse_args()


encoder = pose2img.models.Encoder(in_ch=3)
chainer.serializers.load_npz(args.encoder, encoder)
decoder = pose2img.models.Decoder(out_ch=3)
chainer.serializers.load_npz(args.decoder, decoder)

variable = pose2img.utility.image2variable(args.image)

converted = decoder(encoder(variable, test=True), test=True)

image = pose2img.utility.variable2img(converted)
image.save(args.outfile)
