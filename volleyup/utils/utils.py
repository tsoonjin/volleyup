#!/usr/bin/env python
import cv2
import os
import numpy as np

from feature import FeatureDescriptor
from config import OBJECT_COLOR, RESIZE_FACTOR


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


def get_jpgs(dirpath, skip=0):
    """ Returns all images located in given dirpath
    Parameters
    ----------
    skip : number of frames skip to reduce computation time

    """
    if os.path.exists(os.path.abspath(dirpath)):
        filenames = os.listdir(dirpath)
        # Only attempt to parse and sort files that end with .jpg
        filenames = [filename for filename in filenames if filename.endswith(".jpg")]
        filenames.sort(key=lambda x: int(x.split('.', 1)[0]))
        frames = [cv2.imread('{}/{}'.format(dirpath, filename)) for filename in filenames]
        out = frames[0::skip] if skip > 0 else frames
        print('Read {} images from {}'.format(len(out), dirpath))
        return out
    print('Directory {} does not exist'.format(dirpath))
    return None

def write_jpgs(dirpath, jpgs):
    """ 
    Writes all images to the given dirpath
        
    """
    if os.path.exists(os.path.abspath(dirpath)):
        for i in range(len(jpgs)):
            filename = dirpath + str(i) + ".jpg"
            cv2.imwrite(filename, jpgs[i])
        print('Wrote {} images to {}'.format(len(jpgs), dirpath))
    #print('Directory {} does not exist'.format(dirpath))

# Drawing functions


def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))


def create_windows(names=['original', 'processed']):
    """ Generates windows given list of names """
    for name in names:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)


def workon_frames(frames, func, wait=100):
    """ Wait for n milliseconds before moving on to next frame """
    cv2.namedWindow('original', cv2.WINDOW_NORMAL)
    cv2.namedWindow(func.__name__, cv2.WINDOW_NORMAL)
    for frame in frames:
        cv2.imshow('original', frame)
        cv2.imshow(func.__name__, func(frame))
        k = cv2.waitKey(wait)
        if k == 27:
            break
    cv2.destroyAllWindows()

# Image Analysis


def display_features(frames, channel='gray', feature='sift', wait=100):
    """ Draw keypoints detected using feature supported
    Parameters
    ----------
    feature : feature descriptor used. Supports SIFT, SURF, AKAZE, ORB, BRISK
    channe  : color channel used when computing image feature

    """
    ft = FeatureDescriptor()
    for f in frames:
        kps, descs = ft.compute(get_channel(f, channel), feature)
        ft.draw_features(f, kps)
        cv2.imshow(feature, f)
        k = cv2.waitKey(wait)
        if k == 27:
            break
    cv2.destroyAllWindows()


def display_channels(frames, wait=100):
    " Displayes BGR, HSV and LAB channels information given frames """
    r, c = frames[0].shape[:2]
    for f in frames:
        out = cv2.resize(fuse_channels(f), (int(c * 2), int(r * 2)))
        cv2.imshow('channels', out)
        k = cv2.waitKey(wait)
        if k == 27:
            break
    cv2.destroyAllWindows()


def get_channel(img, channel):
    """ Get specific color channel given an image """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    channels = {'bgr_b': img[..., 0], 'bgr_green': img[..., 1], 'bgr_r': img[..., 2],
                'hsv_h': hsv[..., 0], 'hsv_s': hsv[..., 1], 'hsv_v': hsv[..., 2],
                'lab_l': lab[..., 0], 'lab_a': lab[..., 1], 'lab_b': lab[..., 2],
                'gray': cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)}
    return channels[channel]


def fuse_channels(img):
    """ Returns bgr, hsv and lab channels of image in order """
    return np.vstack((get_bgr_stack(img), get_hsv_stack(img),
                      get_lab_stack(img), get_ycb_stack(img), get_opp_stack(img)))


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


def get_opp_stack(img):
    """ Returns Intensity, R - G, B - Y """
    b, g, r = cv2.split(img)
    I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    R = r - (g+b) / 2
    R.clip(min=0, out=R)
    G = g - (r+b) / 2
    G.clip(min=0, out=G)
    B = b - (r+g) / 2
    B.clip(min=0, out=B)
    Y = ((r+g) / 2) - (np.abs(r - g) / 2) - b
    Y.clip(min=0, out=Y)
    RG = R - G
    RG.clip(min=0, out=RG)
    BY = B - Y
    BY.clip(min=0, out=BY)
    return np.hstack((cv2.cvtColor(I, cv2.COLOR_GRAY2BGR),
                      cv2.cvtColor(RG, cv2.COLOR_GRAY2BGR),
                      cv2.cvtColor(BY, cv2.COLOR_GRAY2BGR)))


def get_salient_stack(img):
    """ Return saliency map for each channels in given image colorspace """
    a, b, c = cv2.split(img)
    a = cv2.cvtColor(get_salient(a), cv2.COLOR_GRAY2BGR)
    b = cv2.cvtColor(get_salient(b), cv2.COLOR_GRAY2BGR)
    c = cv2.cvtColor(get_salient(c), cv2.COLOR_GRAY2BGR)
    return np.hstack((a, b, c))


def get_salient(chan):
    empty = np.ones_like(chan)
    mean = np.mean(chan)
    mean = empty * mean
    blur = cv2.GaussianBlur(chan, (21, 21), 1)
    final = mean - blur
    final = final.clip(min=0)
    final = np.uint8(final)
    return final


# Filters

def get_courtmask(img):
    mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
                       OBJECT_COLOR['court'][0], OBJECT_COLOR['court'][1])
    return mask


def get_playermask(img):
    """ Generates player mask by inverting court mask """
    return cv2.bitwise_not(get_courtmask(img))


def get_netmask(img):
    """ net
        hsv_yellow = cv2.cvtColor(np.uint8([[[ 30,220,230 ]]]), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (hsv_yellow[0][0][0]-15, 100,200), (hsv_yellow[0][0][0]+15, 255, 255))
    """
    """ Sand
    hsv_yellow = cv2.cvtColor(np.uint8([[[ 200,220,230 ]]]), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (hsv_yellow[0][0][0]-15, 0,200), (hsv_yellow[0][0][0]+15, 200, 255))
    """
    """
        Net + logo
    hsv_yellow = cv2.cvtColor(np.uint8([[[ 20,55,120 ]]]), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (hsv_yellow[0][0][0]-20, 150,100), (hsv_yellow[0][0][0]+20, 255, 255))
    """
    """
        back border
    hsv_yellow = cv2.cvtColor(np.uint8([[[ 120,75,140 ]]]), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (hsv_yellow[0][0][0]-15, 120,100), (hsv_yellow[0][0][0]+15, 175, 200))
    """
    """
        front border
    hsv_yellow = cv2.cvtColor(np.uint8([[[ 120,50,140 ]]]), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (hsv_yellow[0][0][0]-20, 70, 100), (hsv_yellow[0][0][0]+20, 175, 200))
    """
    hsv_yellow = cv2.cvtColor(np.uint8([[[ 120,70,140 ]]]), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (hsv_yellow[0][0][0]-15, 50,100), (hsv_yellow[0][0][0]+15, 255, 255))
    img_copy = img.copy()
    res = cv2.bitwise_and(img_copy, img_copy, mask= mask)

    return res


# Image manipulations

def get_roi(img, top_left, bot_right):
    """ Returns region of interest of an img given bounding box points """
    y = [max(top_left[1], 0), min(bot_right[1], img.shape[0] - 1)]
    x = [max(top_left[0], 0), min(bot_right[0], img.shape[1] - 1)]
    return img[y[0]:y[1], x[0]:x[1]]


def resize(img):
    y, x = img.shape[:2]
    return cv2.resize(img, (int(x * RESIZE_FACTOR), int(y * RESIZE_FACTOR)))


def vectorize(img):
    """ Returns image as row vector and shape of original img """
    y, x = img.shape[:2]
    return np.array([img.ravel()]), y, x


def calc_integral(img):
    """ Returns single-channel integral image """
    return cv2.integral(img)

# Color conversion


def normalize_bgr(img):
    b, g, r = cv2.split(np.float32(img))
    total_bgr = b + g + r
    b /= total_bgr
    g /= total_bgr
    r /= total_bgr
    return cv2.merge((cv2.convertScaleAbs(b * 255.0), cv2.convertScaleAbs(g * 255.0),
                      cv2.convertScaleAbs(r * 255.0)))
