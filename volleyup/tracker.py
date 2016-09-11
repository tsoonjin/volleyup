#!/usr/bin/env python
""" Collection of trackers based on different journal papers """
import cv2
import numpy as np

import config

from collections import deque

from utils import get_jpgs, display_channels, display_features


def median_bg_sub(frames, bg_hist=5, thresh_limit=50):
    """ Performs background subtraction via frame differencing with median of previous n frames
    Parameters
    ----------
    bg_hist      : number of frames before current frame considerd as background
    thresh_limit : threshold to differentiate foreground and background

    """
    bg_frames = deque([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames[0:bg_hist]],
                      maxlen=bg_hist)
    for f in frames[bg_hist:]:
        curr = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        diff = np.fabs(curr - np.median(np.array(list(bg_frames)), axis=0))
        mask = cv2.threshold(np.uint8(diff), thresh_limit, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        bg_frames.append(curr)
        cv2.imshow('mask', cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        k = cv2.waitKey(100)
        if k == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    frames = get_jpgs(config.INPUT_IMGS, skip=3)
    display_features(frames, channel='lab_l', feature='orb')
    # median_bg_sub(frames)
