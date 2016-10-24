#!/usr/bin/env python
import sys
import cv2
import signal
from optparse import OptionParser

from tracker.pf.ParticleFilter import ParticleFilter, PlayerParticle
from bg_subtract.bg_subtract import median_bg_sub, eigenbackground
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
    frames = get_jpgs(config.INDVIDUAL_VIDEOS['4'])
    pf = ParticleFilter(PlayerParticle.generate, img_boundary=(frames[0].shape[1], frames[0].shape[0]))
    for img in frames:
        pf.process(img)
        cv2.imshow('particle filter', img)
        k = cv2.waitKey(1)
        if k == 27:
            break


def handle_SIGINT(signal, frame):
    print("Terminated by user")
    exit()

if __name__ == '__main__':
    options = init_env(sys.argv[1:])
    main()
