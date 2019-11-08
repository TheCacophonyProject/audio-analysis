#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2019 Chris Blackbourn

"""For internal use only, see melt.py instead."""

import sys
import time

import common
import ensemble
import squawk


def noise_reduce(file_name):
    import noise_reduction
    sample_rate = 48000
    source = common.load_audio_file_as_numpy_array(file_name, sample_rate)
    nr = noise_reduction.noise_reduce(source, sample_rate)
    return (source, nr, sample_rate)


def find_nr_squawks_from_file_name(file_name):
    (source, nr, sample_rate) = noise_reduce(file_name)
    squawks = squawk.find_squawks(nr, sample_rate)
    return (source, nr, squawks, sample_rate)


def species_identify(source, nr, squawks, sample_rate):
    result = {}
    return result


def speech_detect(source, nr, squawks, sample_rate):
    e = ensemble.ensemble()
    for s in squawks:
        waveform = squawk.extract_squawk_waveform(nr, sample_rate, s)
        e.append_waveform(waveform)
    p = e.apply_model('sd_aa')
    result = {}
    result['speech_detection_version'] = '2019-10-30_A'
    human_squawk_count = 0
    for(pb, ph) in p:
        if pb < 0.1 and ph > 0.95:
            human_squawk_count += 1
    result['speech_detection'] = (human_squawk_count > 3)
    return result


def examine(file_name, summary):
    import cacophony_index
    ci = cacophony_index.calculate(file_name)
    summary.update(ci)
    nss = find_nr_squawks_from_file_name(file_name)
    summary.update(speech_detect(*nss))
    summary.update(species_identify(*nss))


def main():
    argv = sys.argv

    t0 = time.time()
    summary = {}
    result = 0

    if argv[1] == '-cacophony_index':
        import cacophony_index
        ci = cacophony_index.calculate(argv[2])
        summary.update(ci)
    elif argv[1] == '-examine':
        examine(argv[2], summary)
    elif argv[1] == '-noise_reduce':
        (source, nr, sample_rate) = noise_reduce(argv[2])
        common.write_audio_to_file(
            'temp/noise_reduce_stereo.ogg', sample_rate, source, nr)
    elif argv[1] == '-species_identify':
        nss = find_nr_squawks_from_file_name(argv[2])
        summary.update(species_identify(*nss))
    elif argv[1] == '-speech_detect':
        nss = find_nr_squawks_from_file_name(argv[2])
        summary.update(speech_detect(*nss))
    else:
        result = -1

    t1 = time.time()

    summary['processing_time_seconds'] = round(t1 - t0, 1)

    print(common.jsdump(summary))

    return result


if __name__ == '__main__':
    sys.exit(main())
