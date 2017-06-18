# -*- coding: utf-8 -*-
import pose2img
import argparse
import glob
import os
import chainer
import numpy
import time
import pipe
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--image_dataset", required=True)
parser.add_argument("--pose_dataset", required=True)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--outprefix", default="")
args = parser.parse_args()

outdirname = "_".join([
                          args.outprefix,
                          str(int(time.time())),
                      ] | pipe.where(lambda x: len(x) > 0))
OUTPUT_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "output", outdirname)
os.makedirs(OUTPUT_DIRECTORY)
logging.basicConfig(filename=os.path.join(OUTPUT_DIRECTORY, "log.txt"), level=logging.INFO)
console = logging.StreamHandler()
logging.getLogger('').addHandler(console)

image_pathes = glob.glob("{}/*.jpg".format(args.image_dataset))
ids = [os.path.splitext(os.path.basename(path))[0] for path in image_pathes]
pose_pathes = ["{}/{}_rendered.png".format(args.pose_dataset, image_id) for image_id in ids]

image_dataset = pose2img.dataset.ResizedImageDataset(image_pathes, resize=(256, 256))
pose_dataset = pose2img.dataset.ResizedImageDataset(pose_pathes, resize=(256, 256))
dataset = pose2img.dataset.ZippedDataset(image_dataset, pose_dataset)


encoder = pose2img.models.Encoder(in_ch=3)
decoder = pose2img.models.Decoder(out_ch=3)
discriminator = pose2img.models.Discriminator(in_ch=3, out_ch=3)

optimizer_encoder = chainer.optimizers.Adam()
optimizer_encoder.setup(encoder)
optimizer_decoder = chainer.optimizers.Adam()
optimizer_decoder.setup(decoder)
optimizer_discriminator = chainer.optimizers.Adam()
optimizer_discriminator.setup(discriminator)

adversarial_ratio = 0.01

if args.gpu >= 0:
    chainer.cuda.check_cuda_available()
    chainer.cuda.get_device(args.gpu).use()
    encoder.to_gpu()
    decoder.to_gpu()
    discriminator.to_gpu()
    xp = chainer.cuda.cupy
else:
    xp = numpy

iterator = chainer.iterators.SerialIterator(dataset, batch_size=10, repeat=True, shuffle=True)
for i,batch in enumerate(iterator):
    images = chainer.Variable(xp.array([t[0] for t in batch]))
    poses = chainer.Variable(xp.array([t[1] for t in batch]))
    batchsize, _, w, h = poses.data.shape
    encoded = encoder(images)  # type: chainer.Variable
    decoded = decoder(encoded)  # type: chainer.Variable

    y_decoded = discriminator(images, decoded)
    y_true = discriminator(images, poses)

    loss_reconstruction = chainer.functions.mean_absolute_error(
        decoded,
        poses
    )

    loss_adversarial = chainer.functions.sum(chainer.functions.softplus(-y_decoded)) / batchsize / w / h

    optimizer_encoder.zero_grads()
    optimizer_decoder.zero_grads()
    loss_generator = (loss_reconstruction*(1-adversarial_ratio) + loss_adversarial*adversarial_ratio)
    loss_generator.backward()
    optimizer_encoder.update()
    optimizer_decoder.update()

    loss_discriminator_decoded = chainer.functions.sum(chainer.functions.softplus(y_decoded)) / batchsize / w / h
    loss_discriminator_true = chainer.functions.sum(chainer.functions.softplus(-y_true)) / batchsize / w / h
    decoded.unchain_backward()
    poses.unchain_backward()
    optimizer_discriminator.zero_grads()
    loss_discriminator = (loss_discriminator_decoded + loss_discriminator_true)
    loss_discriminator.backward()
    optimizer_discriminator.update()

    save_span = 1000
    report_span = 10
    count_processed = i* batchsize
    if i % report_span == 0:
        logging.info("{}: {} {}".format(count_processed, loss_generator.data, loss_discriminator.data))
    if i % save_span == 0:
        chainer.serializers.save_npz(
            os.path.join(OUTPUT_DIRECTORY, "discriminator_model_{}.npz".format(count_processed)), discriminator)
        chainer.serializers.save_npz(
            os.path.join(OUTPUT_DIRECTORY, "encoder_model_{}.npz".format(count_processed)), encoder)
        chainer.serializers.save_npz(
            os.path.join(OUTPUT_DIRECTORY, "decoder_model_{}.npz".format(count_processed)), decoder)
