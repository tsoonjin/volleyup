#!/usr/bin/env python
""" Vision utiliy module for image processing """
import cv2
import numpy as np
import config


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


def compute_KAZE(img):
    pass

def compute_SIFT(img):
    canvas = img.copy()
    # Use value channel
    v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[2]
    (kps, features) = compute_single_SIFT(v, canvas)
    return canvas


def compute_single_SIFT(channel, canvas):
    """ Compute SIFT features and keypoints given a single channel image """
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(channel, None)
    cv2.drawKeypoints(canvas, kps, canvas, color=config.RED)
    return (kps, features)
