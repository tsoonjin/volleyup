#!/usr/bin/env python
from __future__ import division

import numpy as np
import cv2


# Weight functions

def uniform_weight(N_particles):
    return 1 / N_particles


# Resampling functions

def systematic_resample(particles, weights):
    """ Performs systematic resampling given list of weights
    Inspired by implementation of filterpy by Roger R Labbe Jr
    """
    N = len(weights)

    # make N subdivisions, and choose positions with a consistent random offset
    positions = (np.random.random() + np.arange(N)) / N

    new_particles = []
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            new_particles.append(particles[j])
            i += 1
        else:
            j += 1
    return new_particles


# Transition model

def uniform_displacement(particles, const_dist=5):
    """ Displace particle with const_dist in pixel """
    for p in particles:
        x_old = p.x
        y_old = p.y
        x_new = max(int(x_old + np.random.uniform(-const_dist, const_dist) + gaussian_noise()), 0)
        y_new = max(int(y_old + np.random.uniform(-const_dist, const_dist) + gaussian_noise()), 0)
        p.x = x_new
        p.y = y_new
        p.region = ((int(p.x - p.size[0]/2), int(p.y - p.size[1]/2)),
                    (int(p.x + p.size[0]/2), int(p.y + p.size[1]/2)))


# Noise model

def gaussian_noise(sigma=0.2):
    return np.random.normal(0, sigma)


# Likelihood

def predict_color_hist(particles, img, ref_hist):
    for p in particles:
        y = [max(p.region[0][1], 0), min(p.region[1][1], img.shape[0] - 1)]
        x = [max(p.region[0][0], 0), min(p.region[1][0], img.shape[1] - 1)]
        region = img[y[0]:y[1], x[0]:x[1]]
        p.w = compare_hsv_hist(region, ref_hist)


def compare_hsv_hist(img, ref_hist=None):
    """ Calculates likelihood of particle by measuring histogram similarity using
    Bhattacharyya's distance
    """
    p_hist = hsv_histogram(img)
    return 1 - cv2.compareHist(p_hist, ref_hist, cv2.HISTCMP_BHATTACHARYYA)


# Features

def hue_histgram(img, BIN_SIZE=10):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [BIN_SIZE],
                        [0, 180])
    return cv2.normalize(hist, None).flatten()


def hsv_histogram(img, BIN_SIZE=10):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [BIN_SIZE, BIN_SIZE, BIN_SIZE],
                        [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, None).flatten()
