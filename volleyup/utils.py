#!/usr/bin/env python
import cv2


def get_video_source(filename=None):
    """ Returns video source for processing given filename.
    Uses webcam if no filename is provided """
    source = 'data/{}'.format(filename) if filename else 1
    return cv2.VideoCapture(source)


def create_windows(names=['original', 'processed']):
    """ Generates windows given list of names """
    for name in names:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
