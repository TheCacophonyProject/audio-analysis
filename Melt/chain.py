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

import argparse

NON_BIRD = ["human", "noise"]


def calc_cacophony_index(tracks, length):
    version = "1.0"
    other_labels = [other for other in tracks if other["species"] != "human"]
    bird_percent = 0
    bird_until = -1
    period_length = 20
    bins = math.ceil(length / period_length)
    # some recordings are 61 seconds just make last bin size slightly bigger
    last_bin_size = length - period_length * (bins - 1)
    last_bin = None
    print("last bin is ", last_bin_size)
    if bins > 1 and last_bin_size < 2:
        bins -= 1
        last_bin = length
    percents = []
    for i in range(bins):
        percents.append(
            {
                "begin_s": i * period_length,
                "end_s": min(length, (i + 1) * period_length),
                "index_percent": 0,
            }
        )
    if last_bin is not None:
        percents[-1]["end_s"] = last_bin
    period = 0
    period_length = 20
    if len(percents) > 0:
        period_length = percents[period]["end_s"] - percents[period]["begin_s"]
    period_end = period_length
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
                    period += 1
                    period = min(period, bins - 1)
                    period_length = (
                        percents[period]["end_s"] - percents[period]["begin_s"]
                    )
                    period_end += period_length
            # else:
            bird_percent += new_span[1] - new_span[0]
            # bird_until = new_span[1]
            bird_until = new_span[1]
            period = min(len(percents) - 1, int(bird_until / period_length))
            period = min(period, bins - 1)
            period_length = percents[period]["end_s"] - percents[period]["begin_s"]
    if period < len(percents):
        percents[period]["index_percent"] = round(100 * bird_percent / period_length, 1)

    return percents, version


def filter_tracks(tracks):
    filtered_labels = ["noise"]
    filtered = [t for t in tracks if t["species"] not in filtered_labels]
    return filtered


def species_identify(file_name, morepork_model, bird_models):
    labels = []
    result = {}
    if morepork_model is not None:
        morepork_ids = identify_species(file_name, morepork_model)
        labels.extend(morepork_ids)
    if bird_models is not None:
        for bird_model in bird_models:
            bird_ids, length, chirps = classify(file_name, bird_model)
            bird_ids = filter_tracks(bird_ids)
            labels.extend(bird_ids)
            cacophony_index, version = calc_cacophony_index(bird_ids, length)
            result["cacophony_index"] = cacophony_index
            result["cacophony_index_version"] = version

    result["species_identify"] = labels
    result["species_identify_version"] = "2021-02-01"
    return result


def examine(file_name, morepork_model, bird_models):
    import cacophony_index

    summary = cacophony_index.calculate(file_name)
    summary.update(species_identify(file_name, morepork_model, bird_models))
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old-cacophony-index",
        action="count",
        help="Calculate old cacophony index on this file",
    )
    parser.add_argument("--morepork-model", help="Path to morepork model")
    parser.add_argument("--bird-model", action="append", help="Path to bird model")

    parser.add_argument("file", help="Audio file to run on")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    t0 = time.time()
    summary = None
    result = 0

    if args.old_cacophony_index:
        import cacophony_index

        summary = cacophony_index.calculate(args.file)
    else:
        summary = examine(args.file, args.morepork_model, args.bird_model)

    t1 = time.time()

    summary["processing_time_seconds"] = round(t1 - t0, 1)

    print(common.jsdump(summary))

    return result


if __name__ == "__main__":
    sys.exit(main())
