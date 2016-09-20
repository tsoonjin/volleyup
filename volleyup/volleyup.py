#!/usr/bin/env python
import sys
import cv2
import signal
from optparse import OptionParser

from Video import Video


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
    video = Video('data/beachVolleyball3.mov')
    video.lk_tracker.run()
    cv2.destroyAllWindows()


def handle_SIGINT(signal, frame):
    print("Terminated by user")
    exit()

if __name__ == '__main__':
    options = init_env(sys.argv[1:])
    main()
