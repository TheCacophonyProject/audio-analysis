# -*- coding: utf-8 -*-

# Copyright (C) 2019 Chris Blackbourn

"""Aggregating audio snippets and applying AI models."""

import numpy

import common


def pre_norm_tf(s, scale=1.0):
    target_width = 32768

    adjust_pre = target_width // 2 - s.shape[0] // 2

    if adjust_pre < 0:
        s = s[-adjust_pre:target_width - adjust_pre]
    elif adjust_pre > 0:
        s = numpy.pad(s, (adjust_pre, 0), 'constant')

    adjust_post = target_width - s.shape[0]
    if adjust_post < 0:
        s = s[:adjust_post]
    elif adjust_post > 0:
        s = numpy.pad(s, (0, adjust_post), 'constant')

    s = s.astype(float)

    s = s * common.get_window_const(target_width, 'hamming', scale)

    result = numpy.array(s)
    return result.reshape((target_width, 1))


class ensemble:

    def __init__(self):
        self.xList = []

    def append_waveform(self, waveform):
        self.xList.append(pre_norm_tf(waveform))

    def apply_model(self, flavor):
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
        import tensorflow
        model = tensorflow.keras.models.load_model(
            'model/model_%s.h5' % flavor)
        npx = numpy.array(self.xList)
        return model.predict(npx)
