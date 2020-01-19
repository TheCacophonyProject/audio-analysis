#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2019 Chris Blackbourn

"""Compute the Cacophony Index for an audio file.

The Cacophony Index of an audio file is a measure of the health of the
local ecosystem nearby the recorder.

The Index is a number between 0 and 100, where values less than 50
suggest night time, and values greater than 50 are expected during the daytime.

It works by measuring the change in energy at different frequency bands.
If the energy levels arei changing rapidly it means there is more information in the recording.
More information in the recording normally means a healthier avian ecosystem.

The index is designed to be robust against many types of noise,
such as wind, rain, aircraft engines and other common non-animal sounds.

It assumes the audio is from a recorder outdoors in a natural setting,
away from non-natural sound sources, such as music, motorbikes or talking,
any of which can produce very high Cacophony Index numbers in the 80s and 90s.

Some natural sounds, such as ocean waves or water dripping in a repeated pattern,
will also confuse the index, so care must be taken if abnormally high
cacophony index numbers are reported.

On a technical level, starting with 20 seconds of digital audio sampled at 16kHz,
we slice the audio into 312 overlapping bins. We then use a version of the
FFT algorithm to turn the data into the frequency domain.
The frequency data is further grouped into 10 frequency bands,
somewhat similar to a Mel spectrogram. Energy changes across the bins
are then compared and scored in a robust manner to meet our
background noise design goals.

The Cacophony Index 2019 is a work in progress.
As we continue to collect more data, and our understanding of ecosystem health changes,
from time to time we may update the index to better process the recordings we have,
and provide even better estimates of local ecosystem health.
"""

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


def apply_correction_curve_202001C(raw_score):
    s = raw_score - 10
    return max(100 * s / (s + 18), 0)


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
        if e:
            q = e * (len(points) - bin_20_width) // (entry_count - 1)

        raw_score = score_from_points(points[q:q + bin_20_width])
        score = apply_correction_curve_202001C(raw_score)

        entry = {}
        entry['begin_s'] = round(q * half_window_size / sample_rate)
        entry['end_s'] = round(
            (q + bin_20_width) * half_window_size / sample_rate)
        entry['index_percent'] = round(score, 1)
        table.append(entry)

    result = {}
    result['cacophony_index'] = table
    result['cacophony_index_version'] = '2020-01-20_A'
    if table == []:
        p = source_data.shape[0] / sample_rate
        result['ci_warning'] = 'Cacophony Index requires at least 20 seconds of audio, but only %d seconds of audio were provided.' % p
    return result
