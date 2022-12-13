#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2019 Chris Blackbourn

"""For internal use only, see melt.py instead."""

import sys
import time

import common
from identify_species import identify_species
from identify_bird import classify


def species_identify(file_name, metadata_name, models, bird_model):
    print(
        "values are",
        file_name,
        "meta ",
        metadata_name,
        "models ",
        models,
        "bird model ",
        bird_model,
    )
    labels = identify_species(file_name, metadata_name, models)

    other_labels = classify(file_name, bird_model)
    other_labels = [other for other in other_labels if other["species"] != "human"]
    labels.extend(other_labels)
    result = {}
    result["species_identify"] = labels
    result["species_identify_version"] = "2021-02-01"
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
