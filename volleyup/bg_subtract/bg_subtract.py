#!/usr/bin/env python
import sys
sys.path.append('../')

import cv2
import numpy as np
from collections import deque

from rpca_admm import rpcaADMM
from utils import config
from utils.utils import get_jpgs, workon_frames, resize


# Color ranges
COLOR_RANGE = {'red': [(160, 0, 0), (180, 255, 255)], 'green': [(40, 20, 0), (90, 255, 255)],
               'white': [(0, 0, 200), (180, 30, 255)], 'sand': [(30, 0, 100), (50, 255, 255)]}


def median_bg_sub(frames, bg_hist=10, thresh_limit=40, wait=100):
    """ Performs background subtraction via frame differencing with median of previous n frames
    Parameters
    ----------
    bg_hist      : number of frames before current frame considerd as background
    thresh_limit : threshold to differentiate foreground and background

    """
    name = 'original | motion | background'
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    bg_frames = deque([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames[0:bg_hist]],
                      maxlen=bg_hist)
    for f in frames[bg_hist:]:
        curr = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if len(bg_frames) == bg_hist:
            bg = np.median(np.array(list(bg_frames)), axis=0)
            diff = np.fabs(curr - bg)
            mask = cv2.threshold(np.uint8(diff), thresh_limit, 255, cv2.THRESH_BINARY)[1]
            '''
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
            mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
            '''
            canvas = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cnts = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
            if len(cnts) >= 1:
                cnts.sort(key=cv2.contourArea, reverse=True)
                for cnt in cnts:
                    cnt = cv2.convexHull(cnt)
                    if cv2.contourArea(cnt) > 50:
                        cv2.drawContours(canvas, [cnt], -1, (255, 0, 0), 2)
            out = np.hstack((f, canvas, cv2.cvtColor(np.uint8(bg), cv2.COLOR_GRAY2BGR)))
            cv2.imshow(name, out)
            k = cv2.waitKey(wait)
            bg_frames.clear()
            if k == 27:
                break
        else:
            bg_frames.append(curr)
    cv2.destroyAllWindows()


def eigenbackground(frames):
    y, x = frames[0].shape[:2]
    mean = np.mean([f.ravel() for f in frames], axis=0)
    mean = mean.reshape(y, x, 3)
    cov = np.cov(np.column_stack([cv2.resize(f, (100, 50)).ravel() for f in frames]))
    u, s, v = np.linalg.svd(cov)
    return mean, u, s, v


def rpca_(data):
    res = rpcaADMM(data)
    return cv2.cvtColor(np.uint8(res['X1_admm']), cv2.COLOR_GRAY2BGR)


def gmm_mog2(hist=10, ncomp=16):
    """ Returns Gaussian Mixture Model trained on last hist frames with ncomp mixture components """
    fgbg = cv2.createBackgroundSubtractorMOG2()
    fgbg.setHistory(hist)
    fgbg.setNMixtures(ncomp)
    return fgbg
