# -*- coding: utf-8 -*-
"""
@see https://raw.githubusercontent.com/pfnet-research/chainer-pix2pix/master/net.py
"""
import numpy
import typing

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L


# U-net https://arxiv.org/pdf/1611.07004v1.pdf

# convolution-batchnormalization-(dropout)-relu
class CBR(chainer.Chain):
    def __init__(self, ch0, ch1, bn=True, sample='down', activation=F.relu, dropout=False):
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        layers = {}
        w = chainer.initializers.Normal(0.02)
        if sample == 'down':
            layers['c'] = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        else:
            layers['c'] = L.Deconvolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        if bn:
            layers['batchnorm'] = L.BatchNormalization(ch1)
        super(CBR, self).__init__(**layers)

    def __call__(self, x):
        h = self.c(x)
        if self.bn:
            h = self.batchnorm(h)
        if self.dropout:
            h = F.dropout(h)
        if not self.activation is None:
            h = self.activation(h)
        return h


class Encoder(chainer.Chain):
    def __init__(self, in_ch, n_Layer=8):
        super(Encoder, self).__init__()
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.in_layer = L.Convolution2D(in_ch, 64, 3, 1, 1, initialW=w)
            mid_layers = []
            mid_layers.append(CBR(64, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False))
            mid_layers.append(CBR(128, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False))
            mid_layers.append(CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False))
            mid_layers.append(CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False))
            mid_layers.append(CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False))
            mid_layers.append(CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False))
            if n_Layer == 8:
                mid_layers.append(CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False))
            self.mid_layers = chainer.ChainList(*mid_layers)

    def __call__(self, x):
        hs = [F.leaky_relu(self.in_layer(x))]
        for layer in self.mid_layers:
            hs.append(layer(hs[- 1]))
        return hs


class Decoder(chainer.Chain):
    def __init__(self, out_ch, in_ch=512, will_concat=True, n_Layer=8):
        super(Decoder, self).__init__()
        self.will_concat = will_concat
        w = chainer.initializers.Normal(0.02)
        channel_expansion = 2 if will_concat else 1
        with self.init_scope():
            self.in_layer = CBR(in_ch, 512, bn=True, sample='up', activation=F.relu, dropout=True)
            mid_layers = []
            if n_Layer == 8:
                mid_layers.append(
                    CBR(512 * channel_expansion, 512, bn=True, sample='up', activation=F.relu, dropout=True))
            mid_layers.append(CBR(512 * channel_expansion, 512, bn=True, sample='up', activation=F.relu, dropout=True))
            mid_layers.append(CBR(512 * channel_expansion, 512, bn=True, sample='up', activation=F.relu, dropout=False))
            mid_layers.append(CBR(512 * channel_expansion, 256, bn=True, sample='up', activation=F.relu, dropout=False))
            mid_layers.append(CBR(256 * channel_expansion, 128, bn=True, sample='up', activation=F.relu, dropout=False))
            mid_layers.append(CBR(128 * channel_expansion, 64, bn=True, sample='up', activation=F.relu, dropout=False))
            self.mid_layers = chainer.ChainList(*mid_layers)
            self.out_layer = L.Convolution2D(64 * channel_expansion, out_ch, 3, 1, 1, initialW=w)

    def __call__(self, hs):
        h = self.in_layer(hs[-1])
        for i in range(len(self.mid_layers)):
            if self.will_concat:
                h = F.concat([h, hs[-i - 2]])
            h = self.mid_layers[i](h)
        h = F.concat([h, hs[0]])
        h = self.out_layer(h)
        return h


class NoiseDecoder(Decoder):
    def __init__(self, noise_dimention: int, **kwargs):
        super().__init__(in_ch=(noise_dimention + 512), **kwargs)
        self.noise_dimention = noise_dimention

    def create_noise(self, shape):
        batchsize, ch, width, height = shape
        noise_seed = self.xp.array(self.xp.random.uniform(-1, 1, (batchsize, self.noise_dimention)),
                                   dtype=self.xp.float32)
        noise = chainer.Variable(self.xp.tile(noise_seed, (width, height, 1, 1)).transpose((2, 3, 0, 1)))
        return noise

    def __call__(self, hs, noise=None, *args, **kwargs):
        h = hs[-1]
        if noise is None:
            noise = self.create_noise(h.shape)
        hs_copy = [h_orig for h_orig in hs]
        hs_copy[-1] = chainer.functions.concat(
            (h, noise)
        )
        return super().__call__(hs_copy, *args, **kwargs)


class Discriminator(chainer.Chain):
    def __init__(self, in_ch, out_ch, will_concat=True, layers={}):
        super(Discriminator, self).__init__()
        self.will_concat = will_concat
        channel_expansion = 2 if will_concat else 1
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.c0_0 = CBR(in_ch, 32, bn=False, sample='down', activation=F.leaky_relu, dropout=False)
            self.c0_1 = CBR(out_ch, 32, bn=False, sample='down', activation=F.leaky_relu, dropout=False)
            self.c1 = CBR(32 * channel_expansion, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
            self.c2 = CBR(128, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
            self.c3 = CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
            self.c4 = L.Convolution2D(512, 1, 3, 1, 1, initialW=w)

    def __call__(self, x_0: chainer.Variable, x_1: chainer.Variable):
        h = self.c0_0(x_0)
        if self.will_concat:
            h = F.concat([h, self.c0_1(x_1)])
        h = self.c1(h)
        # hs.append(chainer.functions.average_pooling_2d
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        # h = F.average_pooling_2d(h, h.data.shape[2], 1, 0)
        return h


class BranchDiscriminator(Discriminator):
    def __init__(self, in_ch, out_ch, will_concat=True, layers={}):
        layers['out_1'] = L.Linear(128, 1)
        layers['out_2'] = L.Linear(256, 1)
        super().__init__(in_ch, out_ch, will_concat, layers)

    def __call__(self, x_0: chainer.Variable, x_1: chainer.Variable) -> typing.List[chainer.Variable]:
        hs = []

        h = self.c0_0(x_0)
        if self.will_concat:
            h = F.concat([h, self.c0_1(x_1)])

        h = self.c1(h)
        hs.append(self.out_1(chainer.functions.average_pooling_2d(h, (h.shape[2], h.shape[3]))))
        # hs.append(chainer.functions.average_pooling_2d
        h = self.c2(h)
        hs.append(self.out_2(chainer.functions.average_pooling_2d(h, (h.shape[2], h.shape[3]))))
        h = self.c3(h)
        h = self.c4(h)
        hs.append(h)
        return hs
