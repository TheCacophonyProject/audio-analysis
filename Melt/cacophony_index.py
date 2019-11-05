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
    bass_cut_off_frequency = 100
    bass_cut_off_band = bass_cut_off_frequency * 2 * window_size // sample_rate

    edges = numpy.logspace(
        math.log10(bass_cut_off_band),
        math.log10(window_size),
        num=11,
        dtype=int)

    bins_raw = numpy.split(dct, edges)[1:-1]
    return numpy.array([sum(x * x) for x in bins_raw])


def score_from_points(points):
    points_sorted = sorted(points)
    k0 = int(len(points) * 0.75)
    k1 = int(len(points) * 0.95)
    return 10 * numpy.mean(points_sorted[k0:k1])


def apply_correction_curve_201910B(raw_score):
    s = raw_score - 10
    return 100 * s / (s + 18)


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
    entry_count = (len(points) + 31) // bin_20_width
    for e in range(entry_count):
        q = 0
        if entry_count:
            q = e * (len(points) - bin_20_width) // (entry_count - 1)

        raw_score = score_from_points(points[q:q + bin_20_width])
        score = apply_correction_curve_201910B(raw_score)

        entry = {}
        entry['begin_s'] = round(q * half_window_size / sample_rate)
        entry['end_s'] = round(
            (q + bin_20_width) * half_window_size / sample_rate)
        entry['index_percent'] = round(score, 1)
        table.append(entry)

    result = {}
    result['cacophony_index'] = table
    result['cacophony_index_version'] = '2019-11-05_A'
    if table == []:
        p = source_data.shape[0] / sample_rate
        result['ci_warning'] = 'Cacophony Index requires at least 20 seconds of audio, but only %d seconds of audio were provided.' % p
    return result
