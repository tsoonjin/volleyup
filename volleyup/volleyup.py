#!/usr/bin/env python
import sys
import cv2
from optparse import OptionParser

from utils import get_video_source, create_windows


def init_env(args):
    """ Setup environment and return options """
    # Commandline arguments
    parser = OptionParser(usage="usage: %prog [options...]", prog='volleyup')
    parser.add_option("-t", "--target", dest="target", default=None,
                      help="target video path")
    options = parser.parse_args(args)[0]
    if not options.target:
        print('No input video given')
        exit()

    # GUI setup
    create_windows()
    return options


def main(cap):
    frames = []
    processed = []
    fgbg = cv2.BackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        frames.append(frame)
        fgmask = fgbg.apply(frame)
        processed.append(fgmask)
        cv2.imshow('original', frame)
        cv2.imshow('processed', fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    options = init_env(sys.argv[1:])
    cap = get_video_source(options.target)
    main(cap)
