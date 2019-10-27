# -*- coding: utf-8 -*-

# Copyright (C) 2019 Chris Blackbourn

"""Helper functions for Melt."""

import json
import os
import platform

import numpy


class window_helper:
    cache = {}

    def construct_window(width, family, scale):
        if family == 'bartlett':
            return numpy.bartlett(width) * scale

        if family == 'blackman':
            return numpy.blackman(width) * scale

        if family == 'hamming':
            return numpy.hamming(width) * scale

        if family == 'hanning':
            return numpy.hanning(width) * scale

        if family == 'kaiser':
            beta = 14
            return numpy.kaiser(width, beta) * scale

        if family == 'tukey':
            import scipy.signal
            return scipy.signal.tukey(width) * scale

    def get_window(key):
        if not key in window_helper.cache:
            window_helper.cache[key] = window_helper.construct_window(*key)

        return window_helper.cache[key]


def get_window_const(width, family, scale=1.0):
    return window_helper.get_window((width, family, scale))


def rms(x):
    """Root-Mean-Square."""
    return numpy.sqrt(x.dot(x) / x.size)


def load_audio_file_as_numpy_array(source_file_name, sample_rate):
    import shlex
    import subprocess
    command = 'ffmpeg '
    command += '-i "%s" ' % source_file_name
    command += '-ar %d ' % sample_rate
    command += '-f f32le -c:a pcm_f32le -ac 1 - '

    args = shlex.split(command)
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = p.communicate()

    result = numpy.frombuffer(stdout, dtype=numpy.dtype('<f'))
    return result


def get_os_short_name():
    """Get the short form name of the operating system, either lnx, mac or win."""
    if platform.system() == 'Darwin':
        return 'mac'
    if platform.system() == 'Linux':
        return 'lnx'
    if platform.system() == 'Windows':
        return 'win'
    return 'unknown_platform(%s)' % platform.system()


def get_config_dir():
    return 'venv_' + get_os_short_name()


def get_venv_prefix():
    if get_os_short_name() == 'win':
        return '%s\\Scripts\\activate.bat &&' % get_config_dir()
    return '. %s/bin/activate &&' % get_config_dir()


def execute(command):
    os.system(command)


def jsdump(source):
    return json.dumps(source, sort_keys=True, indent=4, separators=(',', ': '))
