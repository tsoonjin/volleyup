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
