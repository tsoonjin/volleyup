#!/usr/bin/env python
import sys
import cv2
import signal
from optparse import OptionParser

from bg_subtract.bg_subtract import median_bg_sub
from utils import config
from utils.utils import get_jpgs


def init_env(args):
    """ Setup environment and return options """
    # Commandline arguments
    parser = OptionParser(usage="usage: %prog [options...]", prog='volleyup')
    parser.add_option("-t", "--target", dest="target", default=None,
                      help="target video path")
    options = parser.parse_args(args)[0]

    # GUI setup
    return options


def main():
    signal.signal(signal.SIGINT, handle_SIGINT)
    frames = get_jpgs(config.INDVIDUAL_VIDEOS['3'])
    median_bg_sub(frames)


def handle_SIGINT(signal, frame):
    print("Terminated by user")
    exit()

if __name__ == '__main__':
    options = init_env(sys.argv[1:])
    main()
