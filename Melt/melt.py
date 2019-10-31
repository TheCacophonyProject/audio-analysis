#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2019 Chris Blackbourn

"""A multitool for turning raw data, mostly audio, into structured information."""

import sys

import common


def show_help():
    """Displays help."""
    print("""
Melt

> ./melt.py setup
        """)


def setup():
    osn = common.get_os_short_name()
    venv_prefix = common.get_venv_prefix()
    dir_name = common.get_config_dir()
    if osn == 'lnx':
        # if symlink problem:
            # VBoxManage setextradata VM_NAME
            # VBoxInternal2/SharedFoldersEnableSymlinksCreate/FOLDER_NAME 1

        print('sudo apt-get install ffmpeg')
        print('sudo apt-get install python3-venv')
    cmd = 'python3 -m venv %s' % dir_name
    print(cmd)
    cmd = venv_prefix + ' pip install numpy scipy'
    print(cmd)
    ptf = 'tensorflow' if osn == 'mac' else 'tensorflow-gpu'
    cmd = venv_prefix + ' pip install %s' % ptf
    print(cmd)
    print(venv_prefix)


def main():
    argv = sys.argv
    venv_prefix = common.get_venv_prefix()
    dir_name = common.get_config_dir()

    if len(argv) == 1:
        show_help()
        return 0

    if argv[1] == 'setup':
        setup()
        return 0

    suffix = argv[1].split('.')[-1].lower()
    if suffix in '3gp,aac,ac3,adts,aif,aifc,caf,dts,dtshd,flac,gsm,m4a,mp3,mpa,oga,ogg,ra,rif,wav'.split(
            ','):
        cmd = '%s python chain.py -examine %s' % (venv_prefix, argv[1])
        common.execute(cmd)
        return 0

    show_help()
    return -1


if __name__ == '__main__':
    sys.exit(main())
