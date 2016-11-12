#!/usr/bin/env python
import sys
import csv
import cv2
import numpy as np
import signal
from optparse import OptionParser

from tracker.flow import LKTracker
from tracker.pf import ParticleFilter
from tracker.color_tracker import color_tracking
from bg_subtract.bg_subtract import median_bg_sub, eigenbackground, gmm_mog2
from bg_subtract.or_rpca import batch_rpca
from utils import config
from utils.utils import (get_jpgs, workon_frames, normalize_bgr, display_features,
                         display_channels, get_opp_stack)
from utils.preprocess import gamma_correction
from utils.Video import Video


def init_env(args):
    """ Setup environment and return options """
    # Commandline arguments
    parser = OptionParser(usage="usage: %prog [options...]", prog='volleyup')
    parser.add_option("-t", "--target", dest="target", default=3,
                      help="target video path")
    options = parser.parse_args(args)[0]

    # GUI setup
    return options


def main(vid_id=1):
    signal.signal(signal.SIGINT, handle_SIGINT)
    for index in range(5, 6):
        frames = get_jpgs(config.INDVIDUAL_VIDEOS[str(index)])
        pf1, pf2, pf3, pf4 = ParticleFilter.VID_PFS[str(index)]
        csvfile = open('{}beachVolleyball{}_pos.txt'.format('data/csv/', index), 'wb')
        wr = csv.writer(csvfile, csv.QUOTE_NONE)
        for i, f in enumerate(frames[1100:]):
            _, c1 = pf1.process(f)
            _, c2 = pf2.process(f)
            _, c3 = pf3.process(f)
            _, c4 = pf4.process(f)
            print('{}/{} VIDEO{}'.format(i + 1, len(frames), index))
            wr.writerow((c1[0], c1[1], c2[0], c2[1], c3[0], c3[1], c4[0], c4[1]))


def handle_SIGINT(signal, frame):
    print("Terminated by user")
    exit()

if __name__ == '__main__':
    options = init_env(sys.argv[1:])
    main(str(options.target))
