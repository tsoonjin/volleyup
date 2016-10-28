#!/usr/bin/env python
import sys
sys.path.append("../")
import cv2
import numpy as np

from rpca_admm import rpcaADMM
from rpca_alm import robust_pca as rpcaALM
from robust_pcp import pcp
from utils.utils import resize, vectorize, normalize_bgr
from utils.preprocess import gamma_correction
from utils import config


def batch_rpca(frames, output_path='beachVolleyball1'):
    """ Performs batch rpca using ADMM
        @returns: L (low-rank background data), S (sparse foreground outliers)

    """
    y, x = frames[0].shape[:2]
    gray_frames = np.hstack([vectorize(cv2.GaussianBlur(cv2.cvtColor(resize(f), cv2.COLOR_BGR2GRAY),
                                                        (5, 5), 2))[0].T for f in frames])
    res = rpcaADMM(gray_frames)
    L = [cv2.cvtColor(np.uint8(cv2.resize(l.reshape(y / 2, x / 2), (x, y))), cv2.COLOR_GRAY2BGR)
         for l in np.hsplit(res['X3_admm'], len(frames))]
    S = [cv2.cvtColor(np.uint8(cv2.resize(s.reshape(y / 2, x / 2), (x, y))), cv2.COLOR_GRAY2BGR)
         for s in np.hsplit(res['X1_admm'], len(frames))]
    for i, res in enumerate(zip(L, S)):
        cv2.imwrite('{}{}/{}.jpg'.format(config.BG_DIR, output_path, i), res[1])
        cv2.imshow('bg | fg', np.hstack((res[0], res[1])))
        cv2.waitKey(1000)
    cv2.destroyAllWindows()


def online_rpca(M):
    pass
