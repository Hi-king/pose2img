# -*- coding: utf-8 -*-
"""
@see https://raw.githubusercontent.com/pfnet-research/chainer-pix2pix/master/net.py
"""
from __future__ import print_function
import numpy

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
        if sample=='down':
            layers['c'] = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        else:
            layers['c'] = L.Deconvolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        if bn:
            layers['batchnorm'] = L.BatchNormalization(ch1)
        super(CBR, self).__init__(**layers)

    def __call__(self, x, test):
        h = self.c(x)
        if self.bn:
            h = self.batchnorm(h, test=test)
        if self.dropout:
            h = F.dropout(h)
        if not self.activation is None:
            h = self.activation(h)
        return h

class Encoder(chainer.Chain):
    def __init__(self, in_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0'] = L.Convolution2D(in_ch, 64, 3, 1, 1, initialW=w)
        layers['c1'] = CBR(64, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c2'] = CBR(128, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c3'] = CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c4'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c5'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c6'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c7'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        super(Encoder, self).__init__(**layers)

    def __call__(self, x, test=False):
        hs = [F.leaky_relu(self.c0(x))]
        for i in range(1,8):
            hs.append(self['c%d'%i](hs[i-1], test=test))
        return hs

class Decoder(chainer.Chain):
    def __init__(self, out_ch, will_concat=True):
        self.will_concat = will_concat
        layers = {}
        w = chainer.initializers.Normal(0.02)
        channel_expansion = 2 if will_concat else 1
        layers['c0'] = CBR(512, 512, bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c1'] = CBR(512*channel_expansion, 512, bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c2'] = CBR(512*channel_expansion, 512, bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c3'] = CBR(512*channel_expansion, 512, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c4'] = CBR(512*channel_expansion, 256, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c5'] = CBR(256*channel_expansion, 128, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c6'] = CBR(128*channel_expansion, 64, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c7'] = L.Convolution2D(64*channel_expansion, out_ch, 3, 1, 1, initialW=w)
        super(Decoder, self).__init__(**layers)

    def __call__(self, hs, test=False):
        h = self.c0(hs[-1], test=test)
        for i in range(1,8):
            if self.will_concat:
                h = F.concat([h, hs[-i-1]])
            if i<7:
                h = self['c%d'%i](h, test=test)
            else:
                h = self.c7(h)
        return h


class Discriminator(chainer.Chain):
    def __init__(self, in_ch, out_ch, will_concat=True):
        layers = {}
        self.will_concat=will_concat
        channel_expansion = 2 if will_concat else 1
        w = chainer.initializers.Normal(0.02)
        layers['c0_0'] = CBR(in_ch, 32, bn=False, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c0_1'] = CBR(out_ch, 32, bn=False, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c1'] = CBR(32*channel_expansion, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c2'] = CBR(128, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c3'] = CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c4'] = L.Convolution2D(512, 1, 3, 1, 1, initialW=w)
        super(Discriminator, self).__init__(**layers)

    def __call__(self, x_0, x_1, test=False):
        h = self.c0_0(x_0, test=test)
        if self.will_concat:
            h = F.concat([h, self.c0_1(x_1, test=test)])
        h = self.c1(h, test=test)
        h = self.c2(h, test=test)
        h = self.c3(h, test=test)
        h = self.c4(h)
        #h = F.average_pooling_2d(h, h.data.shape[2], 1, 0)
        return h