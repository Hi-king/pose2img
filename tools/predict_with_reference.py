import argparse

import cv2
import numpy
import chainer
import sys
import glob

import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pose2img

SIZE = (128, 64)


def resize_pose_with_reference(image, pose, size):
    h, w = size
    rh, rw = pose.shape[:2]
    ih, iw = image.shape[:2]

    print("pose", pose.shape)
    resized_pose = cv2.resize(pose, (int(rw * float(h) / rh), h))
    # print(image.shape)
    resized_image = cv2.resize(image, (int(iw * float(h) / ih), h))
    print("resized_image", resized_image.shape)

    horizontal = resized_pose.sum(axis=(0, 2))

    h_min, h_max = horizontal.nonzero()[0].min(), horizontal.nonzero()[0].max()
    print("resized_pose:", resized_pose.shape, h_min, h_max)
    # print((h_min + h_max) // 2,  "-", w // 2)
    h_start = min(
        max((h_min + h_max) // 2 - w // 2, 0),
        resized_pose.shape[1] - w
    )
    # print("h_start, ", h_start)
    return resized_image[:, h_start:h_start + w, :], resized_pose[:, h_start:h_start + w, :]


def main(args):
    os.makedirs(args.output_directory, exist_ok=True)
    encoder = pose2img.models.Encoder(in_ch=9, n_Layer=7)
    chainer.serializers.load_npz(args.encoder, encoder)

    decoder = pose2img.models.Decoder(out_ch=3, n_Layer=7)
    chainer.serializers.load_npz(args.decoder, decoder)

    if args.gpu >= 0:
        import cupy
        chainer.cuda.get_device(args.gpu).use()
        encoder.to_gpu(args.gpu)
        decoder.to_gpu(args.gpu)
        xp = cupy
    else:
        xp = numpy

    original_image = pose2img.dataset.Market1501Dataset.read_image(args.reference_image)
    original_pose = pose2img.dataset.Market1501Dataset.read_image(args.reference_pose)
    reference_image, reference_pose = resize_pose_with_reference(original_image, original_pose, SIZE)
    cv2.imwrite("reference_image.png", reference_image[:,:,::-1])
    cv2.imwrite("reference_pose.png", reference_pose[:,:,::-1])
    reference_image, reference_pose = pose2img.dataset.Market1501Dataset.normalize(reference_image), pose2img.dataset.Market1501Dataset.normalize(reference_pose)

    for path in glob.glob(args.input_images):
        print(path)
        target_pose = pose2img.dataset.Market1501Dataset.resize_pose_with_reference(
            reference_image,
            pose2img.dataset.Market1501Dataset.read_image(path),
            need_shift=True
        )
        target_pose = pose2img.dataset.Market1501Dataset.normalize(target_pose)
        input_tensor = numpy.concatenate((reference_image, reference_pose, target_pose), axis=2).transpose(2, 0, 1)
        variable = chainer.Variable(xp.array([input_tensor]))
        converted = decoder(encoder(variable))
        image = pose2img.utility.variable2img(converted)

        image.save(os.path.join(args.output_directory, os.path.basename(path)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--reference_image", required=True)
    parser.add_argument("--reference_pose", required=True)
    parser.add_argument("--input_images", required=True)
    parser.add_argument("--output_directory", required=True)
    parser.add_argument("--gpu", type=int, default=-1)
    args = parser.parse_args()
    main(args)
