import sys
import logging


def init_logging():

    fmt = "%(process)d %(thread)s:%(levelname)7s %(message)s"

    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
    )
