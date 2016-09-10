#!/usr/bin/env python
""" Vision utiliy module for image processing """
import cv2
import numpy as np


def canny_edge(channel, w_diff=1.2):
    """ Performs canny edge detection with automatic thresholding """
    min = np.amin(channel)
    max = np.amax(channel)
    thresh = min + (max - min) / w_diff
    mask = np.uint8(cv2.Canny(channel, thresh / 2, thresh))
    return mask


def intensity_edge_detector(img):
    return cv2.cvtColor(canny_edge(cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[2]),
                        cv2.COLOR_GRAY2BGR)
