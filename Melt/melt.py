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
    """Hints for setting up virtualenv and ffmpeg."""
    osn = common.get_os_short_name()
    venv_prefix = common.get_venv_prefix()
    dir_name = common.get_config_dir()

    if osn == 'lnx':
        # if symlink problem:
            # VBoxManage setextradata VM_NAME
            # VBoxInternal2/SharedFoldersEnableSymlinksCreate/FOLDER_NAME 1

        print('sudo apt-get install ffmpeg')
        print('sudo apt-get install python3-venv')

    if osn == 'mac':
        print('brew install ffmpeg')
        # print('brew install opus-tools') #optional

    if osn == 'win':
        print('Install https://www.ffmpeg.org/download.html')

    cmd = 'python3 -m venv %s' % dir_name
    print(cmd)
    cmd = venv_prefix + ' pip install numpy scipy'
    print(cmd)
    cmd = venv_prefix + ' pip install tensorflow-gpu'
    print(cmd)
    print(venv_prefix)


def pex():
    import os
    #os.system('pip install pex')
    cmd = 'pex . -c pex_entry.py -o melt.pex'
    os.system(cmd)


def main():
    """Main entrypoint for Melt, in normal usage, everything starts here."""

    argv = sys.argv
    venv_prefix = common.get_venv_prefix()
    dir_name = common.get_config_dir()

    if len(argv) == 1:
        show_help()
        return 0

    if argv[1] == 'pex':
        pex()
        return 0

    if argv[1] == 'setup':
        setup()
        return 0

    suffix = argv[1].split('.')[-1].lower()
    exts = '3gp,aac,ac3,adts,aif,aifc,caf,dts,dtshd,flac,gsm,m4a,mp3,mp4,mpa,oga,ogg,opus,ra,rif,wav'

    source_prefix = common.get_source_prefix()
    vpchain_command = '%s python %schain.py ' % (venv_prefix, source_prefix)
    if suffix in exts.split(','):
        cmd = '%s-examine "%s"' % (vpchain_command, argv[1])
        common.execute(cmd)
        return 0

    show_help()
    return -1


if __name__ == '__main__':
    sys.exit(main())
