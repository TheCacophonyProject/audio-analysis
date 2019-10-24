#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2019 Chris Blackbourn

"""Compute the Cacophony Index for an audio file."""

import json
import math
import numpy
import scipy.signal
import sys
import time

import common


def get_ci_bins(source_trim, sample_rate):
    window_size = source_trim.shape[0]  # e.g. 2048
    window_c = common.get_window_const(window_size, 'hanning')
    signal = window_c * source_trim
    dct = scipy.fftpack.dct(signal)
    bassCutOffFreq = 100
    bassCutOffBand = int(bassCutOffFreq * 2 * window_size / sample_rate)

    edges = numpy.logspace(
        math.log10(bassCutOffBand),
        math.log10(window_size),
        num=11,
        dtype=int)
    bins_raw = numpy.split(dct, edges[:-1])
    return numpy.array([sum(x * x) for x in bins_raw[1:]])


def ScoreFromPoints(points):
    pointsSorted = sorted(points)
    k0 = int(len(points) * 0.75)
    k1 = int(len(points) * 0.95)
    result = 10 * numpy.mean(pointsSorted[k0:k1])
    return round(result, 1)


def calculate(source_file_name):
    sample_rate = 16000
    source_data = common.load_audio_file_as_numpy_array(
        source_file_name, sample_rate)

    window_size = 2048

    half_window_size = window_size // 2
    previous_bins = None
    points = []
    for offset in range(
            half_window_size, source_data.shape[0] - half_window_size * 3, half_window_size):
        bins = get_ci_bins(
            source_data[offset:offset + window_size], sample_rate)
        if not previous_bins is None:
            scorePlus = (sum(bins * 2 < previous_bins))
            scoreMinus = (sum(bins > previous_bins * 2))
            points.append(scorePlus + scoreMinus)
        previous_bins = bins

    bin_20_width = 312  # ~20 seconds
    table = []
    for q in range(0, len(points) - bin_20_width, bin_20_width):
        score = ScoreFromPoints(points[q:q + bin_20_width])
        t0 = int(q * half_window_size / sample_rate + 0.5)
        t1 = int((q + bin_20_width) * half_window_size / sample_rate + 0.5)
        entry = {}
        entry['begin_s'] = t0
        entry['end_s'] = t1
        entry['index_percent'] = score
        table.append(entry)

    result = {}
    result['cacophony_index'] = table
    result['cacophony_index_version'] = '2019-10-24_A'
    if table == []:
        p = source_data.shape[0] / sample_rate
        result['ci_warning'] = 'Cacophony Index requires at least 20 seconds of audio, but only %d seconds of audio were provided.' % p
    return result
