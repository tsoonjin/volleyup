#!/usr/bin/env python
""" Collection of trackers based on different journal papers """
import cv2
import numpy as np

import config

from collections import deque

from utils import get_jpgs, display_channels, display_features, workon_frames
from preprocess import gamma_correction, get_court_mask


# Color ranges
COLOR_RANGE = {'red': [(160, 0, 0), (180, 255, 255)], 'green': [(40, 20, 0), (90, 255, 255)],
               'white': [(0, 0, 200), (180, 30, 255)], 'sand': [(30, 0, 100), (50, 255, 255)]}


def median_bg_sub(frames, bg_hist=3, thresh_limit=60):
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
        bg = np.median(np.array(list(bg_frames)), axis=0)
        diff = np.fabs(curr - bg)
        mask = cv2.threshold(np.uint8(diff), thresh_limit, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
        canvas = np.zeros_like(f)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        if len(cnts) >= 1:
            cnts.sort(key=cv2.contourArea, reverse=True)
            for cnt in cnts:
                if cv2.contourArea(cnt) > 50:
                    cv2.drawContours(canvas, [cnt], -1, (255, 0, 0), 2)
        out = np.hstack((f, canvas, cv2.cvtColor(np.uint8(bg), cv2.COLOR_GRAY2BGR)))
        bg_frames.append(curr)
        cv2.imshow(name, out)
        k = cv2.waitKey(1)
        if k == 27:
            break
    cv2.destroyAllWindows()


def cv_bg_subtraction(frames):
    fgbg = cv2.createBackgroundSubtractorMOG2(history=20)
    workon_frames(frames, fgbg.apply)


def color_tracking(frames, color):
    cv2.namedWindow('original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    for f in frames:
        mask = cv2.inRange(cv2.cvtColor(f, cv2.COLOR_BGR2HSV), color[0], color[1])
        cnts = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        players = []
        if len(cnts) >= 1:
            cnts.sort(key=cv2.contourArea, reverse=True)
            for cnt in cnts:
                if len(players) < 2:
                    rect = cv2.minAreaRect(cnt)
                    cv2.drawContours(f, [np.int0(cv2.boxPoints(rect))], -1, (255, 0, 0), 2)
                    players.append(cnt)
                else:
                    break
        cv2.imshow('original', f)
        cv2.imshow('mask', cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        k = cv2.waitKey(100)
        if k == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    frames = get_jpgs(config.INDVIDUAL_VIDEOS['7'])
    median_bg_sub(frames)
