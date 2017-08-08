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
parser.add_argument("--batchsize", type=int, default=10)
parser.add_argument("--adversarial_ratio", type=float, default=0.01)
parser.add_argument("--adversarial_branch", action="store_true")
parser.add_argument("--outprefix", default="")
parser.add_argument("--facade", action="store_true")
parser.add_argument("--random_input", action="store_true")
parser.add_argument("--generator_bias", type=float, default=100)
parser.add_argument("--no_concat_decoder", action="store_false", dest="concat_decoder")
parser.add_argument("--no_concat_discriminator", action="store_false", dest="concat_discriminator")
parser.add_argument("--noise_dimention", default=0, type=int)
args = parser.parse_args()

outdirname = "_".join([
                          args.outprefix,
                          "facade" if args.facade else "",
                          "random_input" if args.random_input else "",
                          "batch" + str(args.batchsize),
                          "ar" + str(args.adversarial_ratio),
                          "generator_bias" + str(args.generator_bias),
                          "no_concat_decoder" if not args.concat_decoder else "",
                          "no_concat_discriminator" if not args.concat_discriminator else "",
                          "noise{}".format(args.noise_dimention) if args.noise_dimention > 0 else "",
                          "branch_adversarial" if args.adversarial_branch else "",
                          str(int(time.time())),
                      ] | pipe.where(lambda x: len(x) > 0))
OUTPUT_DIRECTORY = os.path.join(os.path.dirname(__file__), "output", outdirname)
os.makedirs(OUTPUT_DIRECTORY)
logging.basicConfig(filename=os.path.join(OUTPUT_DIRECTORY, "log.txt"), level=logging.INFO)
console = logging.StreamHandler()
logging.getLogger('').addHandler(console)
logging.info(args)

if args.facade:
    dataset = pose2img.dataset.FacadeDataset(dataDir=args.image_dataset)
else:
    image_pathes_all = glob.glob("{}/*".format(args.image_dataset))
    image_ids = [os.path.splitext(os.path.basename(path))[0] for path in image_pathes_all]
    print(len(image_ids))
    pose_pathes_all = glob.glob("{}/*_rendered.png".format(args.pose_dataset))
    pose_ids = [os.path.basename(path).split("_")[0] for path in pose_pathes_all]
    print(len(pose_ids))

    # use intersection
    ids = list(set(pose_ids).intersection(set(image_ids)))
    pose_pathes = ["{}/{}_rendered.png".format(args.pose_dataset, image_id) for image_id in ids]
    image_pathes = [glob.glob("{}/{}.*".format(args.image_dataset, image_id))[0] for image_id in ids]

    image_dataset = pose2img.dataset.ResizedImageDataset(image_pathes, resize=(286, 286))
    pose_dataset = pose2img.dataset.ResizedImageDataset(pose_pathes, resize=(286, 286))
    dataset = pose2img.dataset.ZippedPreprocessedDataset(
        pose2img.dataset.ZippedDataset(image_dataset, pose_dataset),
        crop=256
    )

if args.gpu >= 0:
    chainer.cuda.check_cuda_available()
    chainer.cuda.get_device(args.gpu).use()
    xp = chainer.cuda.cupy
else:
    xp = numpy

in_channel = 12 if args.facade else 3
encoder = pose2img.models.Encoder(in_ch=in_channel)
if args.noise_dimention > 0:
    decoder = pose2img.models.NoiseDecoder(noise_dimention=args.noise_dimention, out_ch=3,
                                           will_concat=args.concat_decoder)
else:
    decoder = pose2img.models.Decoder(out_ch=3, will_concat=args.concat_decoder)

if args.adversarial_branch:
    discriminator = pose2img.models.BranchDiscriminator(in_ch=in_channel, out_ch=3,
                                                        will_concat=args.concat_discriminator)
else:
    discriminator = pose2img.models.Discriminator(in_ch=in_channel, out_ch=3, will_concat=args.concat_discriminator)


def make_optimizer(model, alpha=0.0002, beta1=0.5):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
    return optimizer


optimizer_encoder = make_optimizer(encoder)
optimizer_decoder = make_optimizer(decoder)
optimizer_discriminator = make_optimizer(discriminator)

adversarial_ratio = args.adversarial_ratio

if args.gpu >= 0:
    encoder.to_gpu()
    decoder.to_gpu()
    discriminator.to_gpu()

iterator = chainer.iterators.SerialIterator(dataset, batch_size=args.batchsize, repeat=True, shuffle=True)


def calc_adversarial_loss(target_variable: chainer.Variable, is_branch: bool, is_positive: bool):
    sign = 1 if is_positive else -1
    if is_branch:
        loss_adversarial = chainer.functions.sum(chainer.functions.softplus(sign * target_variable[0])) / batchsize / w / h
        loss_adversarial += chainer.functions.sum(chainer.functions.softplus(sign * target_variable[1])) / batchsize / w / h
        loss_adversarial += chainer.functions.sum(chainer.functions.softplus(sign * target_variable[2])) / batchsize / w / h
    else:
        loss_adversarial = chainer.functions.sum(chainer.functions.softplus(sign * target_variable)) / batchsize / w / h
    return loss_adversarial

with chainer.using_config('train', True):
    for i, batch in enumerate(iterator):
        if args.random_input:
            images = chainer.Variable(xp.array([numpy.random.random(t[0].shape) for t in batch], dtype=xp.float32))
        else:
            images = chainer.Variable(xp.array([t[0] for t in batch]))
        poses = chainer.Variable(xp.array([t[1] for t in batch]))
        batchsize, _, w, h = poses.data.shape
        encoded = encoder(poses)  # type: chainer.Variable
        decoded = decoder(encoded)  # type: chainer.Variable

        y_decoded = discriminator(poses, decoded)
        y_true = discriminator(poses, images)

        loss_reconstruction = chainer.functions.mean_absolute_error(
            decoded,
            images
        )


        loss_adversarial = calc_adversarial_loss(y_decoded, is_branch=args.adversarial_branch, is_positive=False)
        # loss_adversarial = chainer.functions.sum(chainer.functions.softplus(-y_decoded)) / batchsize / w / h

        encoder.cleargrads()
        decoder.cleargrads()
        loss_generator = args.generator_bias * (
            loss_reconstruction * (1 - adversarial_ratio) + loss_adversarial * adversarial_ratio)
        loss_generator.backward()
        optimizer_decoder.update()
        optimizer_encoder.update()

        loss_discriminator_decoded = calc_adversarial_loss(y_decoded, is_branch=args.adversarial_branch, is_positive=True)
        loss_discriminator_true = calc_adversarial_loss(y_true, is_branch=args.adversarial_branch, is_positive=False)
        decoded.unchain_backward()
        poses.unchain_backward()
        discriminator.cleargrads()
        loss_discriminator = (loss_discriminator_decoded + loss_discriminator_true)
        loss_discriminator.backward()
        optimizer_discriminator.update()

        save_span = 2000
        report_span = 10
        count_processed = i * batchsize
        if i % report_span == 0:
            logging.info("{}: gen={} dis={}".format(count_processed, loss_generator.data, loss_discriminator.data))
            logging.info("{}: adv={} recon={}".format(count_processed, loss_adversarial.data, loss_reconstruction.data))
        if i % save_span == 0:
            with chainer.using_config('train', True):
                converted = decoder(encoder(poses))
            if not args.facade:
                input = pose2img.utility.variable2img(poses)
                input.save(os.path.join(OUTPUT_DIRECTORY, "input_image_{}.png".format(count_processed)))
            image = pose2img.utility.variable2img(converted)
            image.save(os.path.join(OUTPUT_DIRECTORY, "output_image_{}.png".format(count_processed)))

            chainer.serializers.save_npz(
                os.path.join(OUTPUT_DIRECTORY, "discriminator_model_{}.npz".format(count_processed)), discriminator)
            chainer.serializers.save_npz(
                os.path.join(OUTPUT_DIRECTORY, "encoder_model_{}.npz".format(count_processed)), encoder)
            chainer.serializers.save_npz(
                os.path.join(OUTPUT_DIRECTORY, "decoder_model_{}.npz".format(count_processed)), decoder)
