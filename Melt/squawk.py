# -*- coding: utf-8 -*-

# Copyright (C) 2019 Chris Blackbourn

"""Squawk extraction and manipulation."""

import numpy

import common


def paired_item(source):
    source_iter = iter(source)
    while True:
        try:
            yield next(source_iter).item(), next(source_iter).item()
        except StopIteration:
            return


def merge_paired_short_time(udarray, small_time):
    paired_iter = paired_item(udarray)
    r = None
    for s in paired_iter:
        if not r:
            r = s
        elif s[0] < r[1] + small_time:
            r = r[0], s[1]
        else:
            yield r
            r = s
    if r:
        yield r


def find_squawks(source, sample_rate):
    result = []

    source_pad = numpy.pad(source, 1)
    tolerance = common.rms(source) / 3
    t = (abs(source_pad) > tolerance)
    s = numpy.where(numpy.diff(t))[0]
    small_time = int(sample_rate * 0.1)
    for begin_index, end_index in merge_paired_short_time(s, small_time):
        if begin_index + 0.05 * sample_rate < end_index:
            squawk = {'begin_i': begin_index, 'end_i': end_index}
            result.append(squawk)
    return result


def extract_squawk_waveform(source, sample_rate, squawk):
    begin_index = squawk['begin_i']
    end_index = squawk['end_i']
    width = int(0.05 * sample_rate)
    t0 = max(0, begin_index - width)
    t1 = min(source.shape[0], end_index + width)
    result = source[t0:t1]
    if not result.flags['WRITEABLE']:
        result = result.copy()
    result[:begin_index - t0] *= numpy.linspace(0, 1, begin_index - t0)
    result[end_index - t0:t1 - t0] *= numpy.linspace(1, 0, t1 - end_index)
    result *= 0.125 / common.rms(result)
    return result
