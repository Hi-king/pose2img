# -*- coding: utf-8 -*-
import glob
import os
import collections

import cv2
import numpy
import six
import random

import tqdm
from PIL import Image
from chainer.dataset import dataset_mixin


class ZippedDataset(dataset_mixin.DatasetMixin):
    def __init__(self, *bases):
        self.bases = bases
        target_len = len(self.bases[0])
        if any([not len(base) == target_len for base in bases]):
            raise Exception("Length of all datasets should be the same")

    def __len__(self):
        return len(self.bases[0])

    def get_example(self, i):
        return tuple(base[i] for base in self.bases)


class PILImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, paths, resize=None):
        if isinstance(paths, six.string_types):
            with open(paths) as paths_file:
                paths = [path.strip() for path in paths_file]
        self._paths = paths
        self._resize = resize

    def __len__(self):
        return len(self._paths)

    def get_example(self, i) -> Image:
        path = self._paths[i]
        original_image = Image.open(path)
        if not self._resize is None:
            return original_image.resize(self._resize)
        else:
            return original_image


class ResizedImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, paths, resize=None, dtype=numpy.float32):
        self.base = PILImageDataset(paths=paths, resize=resize)
        self._dtype = dtype

    def __len__(self):
        return len(self.base)

    def get_example(self, i) -> numpy.ndarray:
        image = self.base[i]
        image_array = numpy.asarray(image, dtype=self._dtype)
        if len(image_array.shape) == 2:  # gray
            image_array = numpy.stack((image_array, image_array, image_array), axis=-1)

        image_data = image_array.transpose(2, 0, 1)
        if image_data.shape[0] == 4:  # RGBA
            image_data = image_data[:3]
        return image_data


class ZippedPreprocessedDataset(dataset_mixin.DatasetMixin):
    def __init__(self, base: dataset_mixin.DatasetMixin, crop=256):
        self.base = base
        self.crop = crop

    def __len__(self):
        return len(self.base)

    def process_each(self, img, left, top, to_flip: bool):
        cropped = img[:, top:top + self.crop, left:left + self.crop]

        # flip
        if to_flip:
            cropped = cropped[:, :, ::-1]
        return (cropped - 128.0) / 128.0

    def get_example(self, i) -> numpy.ndarray:
        raws = self.base[i]

        _c, height, width = raws[0].shape

        # crop
        left = numpy.random.randint(0, width - self.crop)
        top = numpy.random.randint(0, height - self.crop)
        to_flip = (numpy.random.randint(2) == 0)
        return tuple(self.process_each(img, left, top, to_flip) for img in raws)


class Market1501Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, data_dir):
        """
        :param data_dir: /mnt/dataset/Market-1501/Market-1501-v15.09.15/gt_bbox
        """
        self.pose_dir = data_dir + "_openpose"
        pathdict = collections.defaultdict(list)
        for path in tqdm.tqdm(list(glob.glob("{}/*.jpg".format(data_dir)))):
            basename = os.path.basename(path)
            uid = int(basename.split("_")[0])
            pathdict[uid].append(path)
        self.individuals = list(pathdict.values())

    def __len__(self):
        return len(self.individuals)

    def get_example(self, i):
        list = self.individuals[i]
        random.shuffle(list)
        x, y = list[:2]
        x_item, y_item = self.get_img_pose(x), self.get_img_pose(y)

        input_data = numpy.concatenate((x_item[0], x_item[1], y_item[1]), axis=2)  # image, pose, pose
        target_data = y_item[0]

        return target_data.transpose(2, 0, 1), input_data.transpose(2, 0, 1)

    def read_image(self, path):
        return numpy.asarray(Image.open(path), dtype=numpy.float32)

    def resize_pose_with_reference(self, image, pose):
        h, w = image.shape[:2]
        rh, rw = pose.shape[:2]
        return cv2.resize(pose, (int(rw * float(h) / rh), h))[:, :w, :]

    def normalize(self, data):
        return (data - 128) / 128

    def get_img_pose(self, image_path):
        basename = os.path.basename(image_path)
        pose_path = os.path.join(
            self.pose_dir,
            os.path.splitext(basename)[0] + "_rendered.png")

        image = self.read_image(image_path)
        pose = self.read_image(pose_path)
        pose = self.resize_pose_with_reference(image, pose)
        return self.normalize(image), self.normalize(pose)


class FacadeDataset(dataset_mixin.DatasetMixin):
    def __init__(self, data_dir='./facade/base', data_range=(1, 300)):
        print("load dataset start")
        print("    from: %s" % data_dir)
        print("    range: [%d, %d)" % (data_range[0], data_range[1]))
        self.dataDir = data_dir
        self.dataset = []
        for i in range(data_range[0], data_range[1]):
            img = Image.open(data_dir + "/cmp_b%04d.jpg" % i)
            label = Image.open(data_dir + "/cmp_b%04d.png" % i)
            w, h = img.size
            r = 286 / min(w, h)
            # resize images so that min(w, h) == 286
            img = img.resize((int(r * w), int(r * h)), Image.BILINEAR)
            label = label.resize((int(r * w), int(r * h)), Image.NEAREST)

            img = numpy.asarray(img).astype("f").transpose(2, 0, 1) / 128.0 - 1.0
            label_ = numpy.asarray(label) - 1  # [0, 12)
            label = numpy.zeros((12, img.shape[1], img.shape[2])).astype("i")
            for j in range(12):
                label[j, :] = label_ == j
            self.dataset.append((img, label))
        print("load dataset done")

    def __len__(self):
        return len(self.dataset)

    # return (label, img)
    def get_example(self, i, crop_width=256):
        _, h, w = self.dataset[i][0].shape
        x_l = numpy.random.randint(0, w - crop_width)
        x_r = x_l + crop_width
        y_l = numpy.random.randint(0, h - crop_width)
        y_r = y_l + crop_width
        return self.dataset[i][0][:, y_l:y_r, x_l:x_r], numpy.float32(self.dataset[i][1][:, y_l:y_r, x_l:x_r])
