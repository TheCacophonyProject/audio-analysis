#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2019 Chris Blackbourn

"""For internal use only, see melt.py instead."""

import sys
import time

import common


def main():
    argv = sys.argv

    t0 = time.time()
    summary = {}
    result = -1

    if argv[1] == '-cacophony_index':
        import cacophony_index
        ci = cacophony_index.calculate(argv[2])
        summary.update(ci)
        result = 0

    t1 = time.time()

    summary['speech_detection'] = False
    summary['processing_time_seconds'] = round(t1 - t0, 1)

    print(common.jsdump(summary))

    return result


if __name__ == '__main__':
    sys.exit(main())
