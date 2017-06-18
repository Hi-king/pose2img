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
        image_data = numpy.asarray(image, dtype=self._dtype).transpose(2, 0, 1)
        if image_data.shape[0] == 4:  # RGBA
            image_data = image_data[:3]
        return image_data


class PreprocessedDataset(dataset_mixin.DatasetMixin):
    def __init__(self, base: dataset_mixin.DatasetMixin):
        self.base = base

    def __len__(self):
        return len(self.base)

    def get_example(self, i) -> numpy.ndarray:
        raw = self.base[i]
        if numpy.random.randint(2) == 0:
            raw = raw[:, :, ::-1]
        return (raw - 128.0) / 128.0
