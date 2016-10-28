#!/usr/bin/env python
import sys
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
    signal.signal(signal.SIGINT, handle_SIGINT)
    frames = get_jpgs(config.INDVIDUAL_VIDEOS[vid_id], skip=5)
    video = Video('{}.mov'.format(config.INDVIDUAL_VIDEOS['3']))
    pf_white_far = ParticleFilter(PlayerParticle.generate, ref_img=white_far,
                                  img_boundary=(frames[0].shape[1], frames[0].shape[0]))
    pf_green_near = ParticleFilter(PlayerParticle.generate, ref_img=green_near,
                                   img_boundary=(frames[0].shape[1], frames[0].shape[0]))
    pf_green_far = ParticleFilter(PlayerParticle.generate, ref_img=green_far,
                                  img_boundary=(frames[0].shape[1], frames[0].shape[0]))
    display_channels(frames)
    # display_features(frames, channel='gray', feature='orb')


def handle_SIGINT(signal, frame):
    print("Terminated by user")
    exit()

if __name__ == '__main__':
    options = init_env(sys.argv[1:])
    main(str(options.target))
