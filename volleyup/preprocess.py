#!/usr/bin/env python
""" Module to handle preprocessing of images """
import cv2
import numpy as np


def gamma_correction(img, gamma=2.2):
    """ Normalizes illumination for colored image
    Parameters
    ----------
    gamma : value < 1 makes image darker lighter otherwise

    """
    img = np.float32(img)
    img /= 255.0
    img = cv2.pow(img, 1 / gamma) * 255
    img = np.uint8(img)
    return img


def get_court_mask(img, thresh=50):
    """ Returns mask of object with high hue which eliminated audiences from image """
    hue = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[0]
    return cv2.threshold(hue, thresh, 255, cv2.THRESH_BINARY_INV)[1]
