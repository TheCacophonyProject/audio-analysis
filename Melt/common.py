# -*- coding: utf-8 -*-

# Copyright (C) 2019 Chris Blackbourn

"""Helper functions for Melt."""

import json
import os
import platform

import numpy


def check_python_version():
    import sys
    if sys.version_info[0] < 3:
        print('python version 2 not supported, try activate virtualenv or run setup.')
        sys.exit()


class window_helper:
    cache = {}

    def construct_window(width, family, scale):
        if family == 'bartlett':
            return numpy.bartlett(width) * scale

        if family == 'blackman':
            return numpy.blackman(width) * scale

        if family == 'hamming':
            return numpy.hamming(width) * scale

        if family == 'hann':
            import scipy.signal
            return scipy.signal.hann(width) * scale

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
    check_python_version()
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


def bytesio_from_audio(sample_rate, source_left, source_right):
    import io
    import wave

    bio = io.BytesIO()
    w = wave.open(bio, 'wb')
    w.setsampwidth(2)
    w.setframerate(sample_rate)
    if source_right is None:
        w.setnchannels(1)
        source = source_left
    else:
        w.setnchannels(2)
        slr = (source_left, source_right)
        source = numpy.stack(slr, axis=1)
    data = (32768 * source)
    data = numpy.clip(data, -32768, 32767)
    data = data.astype('<h')
    w.writeframesraw(data.tostring())
    w.close()
    bio.seek(0)
    return bio


def write_audio_to_file(file_name, sample_rate,
                        source_left, source_right=None):
    import shlex
    import subprocess

    if file_name.endswith('.ogg') and source_right is None:
        # ffmpeg vorbis encoder only stereo
        source_right = source_left

    bio = bytesio_from_audio(sample_rate, source_left, source_right)

    command = None
    if file_name.endswith('.ogg'):
        command = 'ffmpeg -y -i - -c:a vorbis -strict -2 %s' % file_name

    if command:
        print(command)
        p = subprocess.Popen(
            shlex.split(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        p.communicate(input=bio.read())
    else:
        with open(file_name, 'wb') as f:
            f.write(bio.read())


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
