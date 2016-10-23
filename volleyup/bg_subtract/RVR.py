#!/usr/bin/env python
import cv2
import numpy as np

from utils import get_jpgs, resize
from Video import Video
from config import INDVIDUAL_VIDEOS
from rpcaADMM import rpcaADMM

""" Implements
1) Robust Video Registration Applied to Field-Sports Video Analysis
https://ivul.kaust.edu.sa/Documents/Publications/2012/Robust%20Video%20Registration%20Applied%20to%20Field-Sports%20Video%20Analysis.pdf
2) Automatic Recognition of Offensive Team Formation in American Football
http://vision.ai.illinois.edu/publications/atmosukarto_cvsports13.pdf
"""


def img_diff(img_src, img_dst, h):
    """ Calculates pixel-wise difference after applying homography to base img
    @param homography: 1 x 8 vector assuming 8 DOF

    """
    return img_dst - np.dot(img_src, h)


def jacobian(img_src, h):
    pass


def vectorize_img(img):
    return img.ravel()


def homography(img_src, img_dst, reproj_thresh=4.0):
    src_pts = np.float32(img_src).reshape(-1, 1, 2)
    dst_pts = np.float32(img_dst).reshape(-1, 1, 2)
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
    print(H)
    return H


def rvr(frames):
    res = []
    for frame in frames:
        h = rpcaADMM(cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 2))
        res.append(h)
    return res


def process_video(video_num, wait=1):
    frames = []
    vid = Video('{}.mov'.format(INDVIDUAL_VIDEOS[str(video_num)]))
    cap = vid.cap
    while True:
        ret, frame = cap.read()
        # Operates on grayscale frame
        if vid.is_eov():
            break
        frames.append(frame)
        cv2.imshow('original', frame)
        k = cv2.waitKey(wait)
        if k == 27:
            print("Terminated by user")
            exit()
    cap.release()
    cv2.destroyAllWindows()
    return frames


if __name__ == '__main__':
    frames = process_video(3, wait=10)
    frames = get_jpgs(INDVIDUAL_VIDEOS['3'], skip=10)
    frames = [resize(f) for f in frames]
