#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2019 Chris Blackbourn

"""For internal use only, see melt.py instead."""

import sys
import time

import common
from identify_species import identify_species
from identify_bird import classify
import math


NON_BIRD = ["human", "noise"]


def calc_cacophony_index(tracks, length):
    version = "1.0"
    other_labels = [other for other in tracks if other["species"] != "human"]
    bird_percent = 0
    bird_until = -1
    period_length = 20
    bins = math.ceil(length / period_length)

    # some recordings are 61 seconds just make last bin size slightly bigger
    last_bin_size = period_length - period_length * (bins - 1)
    if last_bin_size < 2:
        bins -= 1
    percents = []
    for i in range(bins):
        percents.append(
            {
                "begin_s": i * period_length,
                "end_s": (i + 1) * period_length,
                "index_percent": 0,
            }
        )
    period_end = period_length
    period = 0
    for track in other_labels:
        if track["species"] not in NON_BIRD:
            # bird started in existing span
            if bird_until >= track["begin_s"] and bird_until < track["end_s"]:
                new_span = (bird_until, track["end_s"])
            # bird started after current span
            elif bird_until < track["end_s"]:
                new_span = (track["begin_s"], track["end_s"])
            else:
                continue
            if new_span[1] > period_end:
                while new_span[1] > period_end:
                    if new_span[0] < period_end:
                        bird_percent += period_end - new_span[0]
                        new_span = (period_end, new_span[1])
                        # bird_percent = min(period_length, new_span[1] - period_end)
                    percents[period]["index_percent"] = round(
                        100 * bird_percent / period_length, 1
                    )

                    bird_percent = 0
                    period_end += period_length
                    period += 1
                    period = min(period, bins - 1)

            # else:
            bird_percent += new_span[1] - new_span[0]
            # bird_until = new_span[1]
            bird_until = new_span[1]
            period = min(len(percents) - 1, int(bird_until / period_length))
            period = min(period, bins - 1)
    if period < len(percents):
        percents[period]["index_percent"] = round(100 * bird_percent / period_length, 1)

    return percents, version


def filter_trcks(tracks):
    filtered_labels = ["noise"]
    filtered = [t for t in tracks if t["species"] not in filtered_labels]
    return filtered


def species_identify(file_name, metadata_name, models, bird_model):
    labels = identify_species(file_name, metadata_name, models)
    other_labels, length = classify(file_name, bird_model)
    other_labels = filter_trcks(other_labels)
    cacophony_index, version = calc_cacophony_index(other_labels, length)

    labels.extend(other_labels)
    result = {}
    result["species_identify"] = labels
    result["species_identify_version"] = "2021-02-01"
    result["cacophony_index"] = cacophony_index
    result["cacophony_index_version"] = version
    return result


def examine(file_name, metadata_name, models, other_model, summary):
    import cacophony_index

    ci = cacophony_index.calculate(file_name)
    summary.update(ci)
    summary.update(species_identify(file_name, metadata_name, models, other_model))


def main():
    argv = sys.argv

    t0 = time.time()
    summary = {}
    result = 0

    if argv[1] == "-cacophony_index":
        import cacophony_index

        ci = cacophony_index.calculate(argv[2])
        summary.update(ci)
    else:
        examine(argv[1], argv[2], argv[3], argv[4], summary)

    t1 = time.time()

    summary["processing_time_seconds"] = round(t1 - t0, 1)

    print(common.jsdump(summary))

    return result


if __name__ == "__main__":
    sys.exit(main())
