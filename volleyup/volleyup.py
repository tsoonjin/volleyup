#!/usr/bin/env python
import sys
import signal
import cv2
from optparse import OptionParser

from balltracker.bgsub_stand_homo import BallFinder
from tracker.pf import ParticleFilter
from utils import config
from utils.utils import (get_jpgs, workon_frames, display_channels, get_opp_stack)


def init_env(args):
    """ Setup environment and return options """
    # Commandline arguments
    parser = OptionParser(usage="usage: %prog [options...]", prog='volleyup')
    parser.add_option("-t", "--target", dest="target", default=3,
                      help="target video path")
    options = parser.parse_args(args)[0]

    # GUI setup
    return options


def main(vid_id='1'):
    # Initializes Ballfinder
    signal.signal(signal.SIGINT, handle_SIGINT)
    for i in range(1, 8):
        vid = cv2.VideoCapture('data/topdown_vids/topdown{}.avi'.format(i))
        frames = []
        while True:
            ret, frame = vid.read()
            frames.append(frame)
            print(len(frames))
            cv2.imwrite('data/topdown_frames/beachVolleyball{}/{}.jpg'.format(i, len(frames)), frame)
            if vid.get(cv2.CAP_PROP_POS_FRAMES) == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                break
            cv2.imshow('stitched', frame)
            k = cv2.waitKey(1)
            if k == 27:
                print("Terminated by user")
                exit()
        vid.release()
        cv2.destroyAllWindows()
    '''
    # Processing all frames
    # for index in range(1, 8):
    frames = get_jpgs(config.INDVIDUAL_VIDEOS[vid_id], skip=5)
    # Independent particle filter for each player
    pfs = ParticleFilter.VID_PFS[vid_id]
    # path = '{}beachVolleyball{}_pos.txt'.format('data/csv/', index)
    ParticleFilter.track_players(frames, pfs, debug=True)
    '''


def handle_SIGINT(signal, frame):
    print("Terminated by user")
    exit()

if __name__ == '__main__':
    options = init_env(sys.argv[1:])
    main(str(options.target))
