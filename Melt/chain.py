#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2019 Chris Blackbourn

"""For internal use only, see melt.py instead."""

import sys
import time

import common
from identify_morepork import identify_morepork
from identify_tracks import classify, get_max_chirps, NON_BIRD
import math
from pathlib import Path
import argparse
import json
import logging


def calc_cacophony_index(tracks, length):
    version = "1.0"
    bird_percent = 0
    bird_until = -1
    period_length = 20
    bins = math.ceil(length / period_length)
    # some recordings are 61 seconds just make last bin size slightly bigger
    last_bin_size = length - period_length * (bins - 1)
    last_bin = None
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
    for track in tracks:
        # bird started in existing span
        if bird_until >= track.start and bird_until < track.end:
            new_span = (bird_until, track.end)
        # bird started after current span
        elif bird_until < track.end:
            new_span = (track.start, track.end)
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
                period_length = percents[period]["end_s"] - percents[period]["begin_s"]
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
    filtered = [
        t
        for t in tracks
        if len(t.predictions[0].labels) > 0
        and any([l for l in t.predictions[0].labels if l not in NON_BIRD])
    ]
    return filtered


def species_identify(file_name, morepork_model, bird_models, analyse_tracks):
    labels = []
    result = {}
    meta_file = Path(file_name).with_suffix(".txt")
    meta_data = None
    region_code = None
    if meta_file.exists():
        with meta_file.open("r") as f:
            meta_data = json.load(f)
    if morepork_model is not None and not analyse_tracks:
        morepork_ids = identify_morepork(file_name, morepork_model)
        labels.extend(morepork_ids)
    if bird_models is not None:
        classify_res = classify(file_name, bird_models, analyse_tracks, meta_data)
        if classify_res is not None:
            tracks, length, chirps, signals, raw_length = classify_res
            cacophony_index, version = calc_cacophony_index(
                filter_tracks(tracks), length
            )
            if meta_data is not None:
                observed_species, region_code = species_by_location(meta_data)
                if region_code is not None:
                    logging.debug(
                        "Matching to region code %s species list %s",
                        region_code,
                        observed_species,
                    )
                    for track in tracks:
                        for prediction in track.predictions:
                            if len(prediction.ebird_ids) == 0:
                                continue
                            t_labels = []
                            ebird_ids = []
                            for label, pred_ebird_ids in zip(
                                prediction.labels, prediction.ebird_ids
                            ):
                                found = len(pred_ebird_ids) == 0 or next(
                                    (
                                        True
                                        for track_ebird in pred_ebird_ids
                                        if track_ebird in observed_species
                                    ),
                                    False,
                                )
                                if found:
                                    t_labels.append(label)
                                    ebird_ids.append(pred_ebird_ids)
                                else:
                                    logging.debug("Region filtering %s", label)
                                    prediction.filtered_labels.append(
                                        (label, pred_ebird_ids)
                                    )

                            prediction.labels = t_labels
                            prediction.ebird_ids = ebird_ids
            labels.extend([track.get_meta() for track in tracks])

            if not analyse_tracks:
                max_chirps = get_max_chirps(length)
                version = "2.0"
                chirp_index = 0 if max_chirps == 0 else round(100 * chirps / max_chirps)
                if region_code is not None:
                    result["region_code"] = region_code
                result["duration"] = raw_length
                result["cacophony_index"] = cacophony_index
                result["cacophony_index_version"] = version
                result["chirps"] = {
                    "chirps": chirps,
                    "max_chirps": max_chirps,
                    "chirp_index": chirp_index,
                    "signals": [s.to_array() for s in signals],
                }

    result["species_identify"] = labels
    result["species_identify_version"] = "2021-02-01"
    return result


def species_by_location(rec_metadata):
    species_file = Path("/Melt/ebird_species.json")
    if species_file.exists():
        with species_file.open("r") as f:
            species_data = json.load(f)
    else:
        logging.info("No species file")
        return None, None
    location_data = rec_metadata.get("location")
    species_list = set()
    region_code = None
    if location_data is None:
        region_code = "NZ"
        logging.info("No location data assume nz species")
        for species_info in species_data.values():
            region_info = species_info["region"]["info"]
            parent_info = region_info.get("parent")
            if (
                region_info["type"] == "country" and region_info["code"] == region_code
            ) or (parent_info is not None and parent_info["code"] == region_code):
                species_list.update(species_info["species"])
        species_list = list(species_list)
    else:
        lat = location_data.get("lat")
        lng = location_data.get("lng")
        # "lat": -36.997142,
        # "lng": 174.57328

        # match lat lng
        for code, species_info in species_data.items():
            region_bounds = species_info["region"]["info"]["bounds"]
            if (
                lng >= region_bounds["minX"]
                and lng <= region_bounds["maxX"]
                and lat >= region_bounds["minY"]
                and lat <= region_bounds["maxY"]
            ):
                species_list = species_info["species"]
                region_code = code
                logging.info(
                    "Match lat %s lng %s to region %s ", lat, lng, species_info
                )
                break

                #         "minX": 174.160829,
                # "maxX": 175.551667,
                # "minY": -37.380549,
                # "maxY": -35.899166
    return species_list, region_code


def examine(file_name, morepork_model, bird_model, analyse_tracks=False):
    import cacophony_index

    summary = cacophony_index.calculate(file_name)
    summary.update(
        species_identify(file_name, morepork_model, bird_model, analyse_tracks)
    )
    return summary


def none_or_str(value):
    if value.lower() in ["none", "null"]:
        return None
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--meta-to-stdout",
        action="count",
        help="Print metadata to stdout instead of saving to file.",
    )
    parser.add_argument(
        "--old-cacophony-index",
        action="count",
        help="Calculate old cacophony index on this file",
    )
    parser.add_argument(
        "--morepork-model",
        default=None,
        type=none_or_str,
        help="Path to morepork model",
    )
    parser.add_argument(
        "--bird-model",
        # default=["/models/bird-model"],
        type=none_or_str,
        action="append",
        help="Path to bird model",
    )

    parser.add_argument("file", help="Audio file to run on")

    parser.add_argument(
        "--analyse-tracks",
        type=str2bool,
        default=False,
        help="Classify human made tracks marked with classify flag, in metadata file",
    )

    args = parser.parse_args()
    if args.bird_model is None or len(args.bird_model) == 0:
        args.bird_model = ["/models/bird-model/audioModel.keras"]

    return args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    args = parse_args()
    init_logging()
    t0 = time.time()
    summary = None
    result = 0

    if args.old_cacophony_index:
        import cacophony_index

        summary = cacophony_index.calculate(args.file)
    else:
        summary = examine(
            args.file,
            args.morepork_model,
            args.bird_model,
            analyse_tracks=args.analyse_tracks,
        )

    t1 = time.time()

    summary["processing_time_seconds"] = round(t1 - t0, 1)
    if args.meta_to_stdout:
        print(common.jsdump(summary))
    else:
        audio_file = Path(args.file)
        metadata_file = audio_file.with_suffix(".txt")
        logging.info("Writing metadata to %s", metadata_file)

        if metadata_file.exists():
            with metadata_file.open("r") as f:
                metadata = json.load(f)
        else:
            metadata = {}
        metadata["analysis_result"] = summary
        with metadata_file.open("w") as f:
            json.dump(metadata, f, sort_keys=True, indent=4)

    return result


def init_logging():

    fmt = "%(process)d %(thread)s:%(levelname)7s %(message)s"

    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
    )


if __name__ == "__main__":
    try:
        main()
    except:
        logging.error("Terminated with error", exc_info=True)
        sys.exit(1)
