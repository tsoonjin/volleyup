#!/usr/bin/env python
import cv2
import os
import glob
import numpy as np

import config


# Input Output

def get_basename(name):
    path = os.path.abspath(name)
    return os.path.basename(os.path.normpath(path))


def get_video_source(filename=None):
    """ Returns video source for processing given filename.
    Uses webcam if no filename is provided """
    source = filename if filename else 1
    if not os.path.isfile(os.path.abspath(source)):
        print("File {} not found".format(os.path.abspath(source)))
        exit()
    return cv2.VideoCapture(source)


def get_jpgs(dirpath, skip=60):
    """ Returns all images located in given dirpath
    Parameters
    ----------
    skip : number of frames skip to reduce computation time

    """
    if os.path.exists(os.path.abspath(dirpath)):
        filenames = os.listdir(dirpath)
        filenames.sort(key=lambda x: int(x.split('.', 1)[0]))
        frames = [cv2.imread('{}/{}'.format(dirpath, filename))
                  for filename in filenames if filename.endswith('.jpg')]
        out = frames[0::skip]
        print('Read {} images from {}'.format(len(out), dirpath))
        return out
    print('Directory {} does not exist'.format(dirpath))
    return None

# Drawing functions


def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))


def create_windows(names=['original', 'processed']):
    """ Generates windows given list of names """
    for name in names:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)


def workon_frames(frames, wait=1000):
    """ Wait for n milliseconds before moving on to next frame """
    for frame in frames:
        cv2.imshow('original', frame)
        k = cv2.waitKey(wait)
        if k == 27:
            break
    cv2.destroyAllWindows()

# Image Analysis


def get_channels(img):
    """ Returns bgr, hsv and lab channels of image in order """
    return np.vstack((get_bgr_stack(img), get_hsv_stack(img), get_lab_stack(img)))


def get_bgr_stack(img):
    """ Returns horizontal stack of BGR channels """
    b, g, r = cv2.split(img)
    b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    r = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
    return np.hstack((b, g, r))


def get_hsv_stack(img):
    return get_bgr_stack(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))


def get_luv_stack(img):
    return get_bgr_stack(cv2.cvtColor(img, cv2.COLOR_BGR2LUV))


def get_ycb_stack(img):
    return get_bgr_stack(cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB))


def get_lab_stack(img):
    return get_bgr_stack(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
