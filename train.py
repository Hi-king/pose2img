# -*- coding: utf-8 -*-
import pose2img
import argparse
import glob
import os
import chainer
import numpy

parser = argparse.ArgumentParser()
parser.add_argument("--image_dataset", required=True)
parser.add_argument("--pose_dataset", required=True)
args = parser.parse_args()

image_pathes = glob.glob("{}/*.jpg".format(args.image_dataset))
ids = [os.path.splitext(os.path.basename(path))[0] for path in image_pathes]
pose_pathes = ["{}/{}_rendered.png".format(args.pose_dataset, image_id) for image_id in ids]

image_dataset = pose2img.dataset.ResizedImageDataset(image_pathes, resize=(240, 320))
pose_dataset = pose2img.dataset.ResizedImageDataset(pose_pathes, resize=(240, 320))
dataset = pose2img.dataset.ZippedDataset(image_dataset, pose_dataset)

iterator = chainer.iterators.SerialIterator(dataset, batch_size=10, repeat=True, shuffle=True)
for batch in iterator:
    images = chainer.Variable(numpy.array([t[0] for t in batch]))
    poses = chainer.Variable(numpy.array([t[1] for t in batch]))
    print(len(batch))
    print(images.data.shape)
    print(poses.data.shape)
    exit(0)
