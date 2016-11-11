#!/usr/bin/env python
import sys
import csv
import cv2
import numpy as np
import signal
from optparse import OptionParser

from tracker.flow import LKTracker
from tracker.pf.ParticleFilter import ParticleFilter, PlayerParticle
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
    white_far = cv2.imread('{}bra_white_far1.png'.format(config.TEMPLATE_DIR))
    green_near = cv2.imread('{}lat_green_near1.png'.format(config.TEMPLATE_DIR))
    green_far = cv2.imread('{}lat_green_far1.png'.format(config.TEMPLATE_DIR))
    red_near = cv2.imread('{}usa_red_near5.png'.format(config.TEMPLATE_DIR))
    red_far = cv2.imread('{}usa_red_far5.png'.format(config.TEMPLATE_DIR))
    signal.signal(signal.SIGINT, handle_SIGINT)
    frames = get_jpgs(config.INDVIDUAL_VIDEOS[vid_id])
    pf1 = ParticleFilter(PlayerParticle.generate, ref_img=white_far, color=config.BRAZIL['color'],
                         img_boundary=(frames[0].shape[1], frames[0].shape[0]),
                         size=(30, 40))
    pf2 = ParticleFilter(PlayerParticle.generate, ref_img=white_far, color=config.BRAZIL['color'],
                         img_boundary=(frames[0].shape[1], frames[0].shape[0]),
                         size=(30, 40))
    pf3 = ParticleFilter(PlayerParticle.generate, ref_img=green_far, color=config.LATVIA['color'],
                         img_boundary=(frames[0].shape[1], frames[0].shape[0]),
                         size=(30, 60))
    pf4 = ParticleFilter(PlayerParticle.generate, ref_img=green_near, color=config.LATVIA['color'],
                         img_boundary=(frames[0].shape[1], frames[0].shape[0]))

    centers = []
    for i, f in enumerate(frames):
        print(i + 1)
        _, c1 = pf1.process(f)
        _, c2 = pf2.process(f)
        _, c3 = pf3.process(f)
        _, c4 = pf4.process(f)
        centers.append((c1[0], c1[1], c2[0], c2[1], c3[0], c3[1], c4[0], c4[1]))
        print(centers)
        # cv2.imshow('res', f)
        # cv2.waitKey(1)

    with open('data/csv/beachVolleyball1.csv', 'wb') as csvfile:
        wr = csv.writer(csvfile, csv.QUOTE_NONE)
        for c in centers:
            wr.writerow(c)


def handle_SIGINT(signal, frame):
    print("Terminated by user")
    exit()

if __name__ == '__main__':
    options = init_env(sys.argv[1:])
    main(str(options.target))
