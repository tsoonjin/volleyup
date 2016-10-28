#!/usr/bin/env python
""" Module to handle preprocessing of images """
import cv2
import numpy as np


def contrast_enhance(img):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(3, 3))
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    return cv2.cvtColor(cv2.merge((h, s, clahe.apply(v))), cv2.COLOR_HSV2BGR)


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
