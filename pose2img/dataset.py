# -*- coding: utf-8 -*-
import os

import numpy
import six
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


class FacadeDataset(dataset_mixin.DatasetMixin):
    def __init__(self, dataDir='./facade/base', data_range=(1, 300)):
        print("load dataset start")
        print("    from: %s" % dataDir)
        print("    range: [%d, %d)" % (data_range[0], data_range[1]))
        self.dataDir = dataDir
        self.dataset = []
        for i in range(data_range[0], data_range[1]):
            img = Image.open(dataDir + "/cmp_b%04d.jpg" % i)
            label = Image.open(dataDir + "/cmp_b%04d.png" % i)
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
