#!/usr/bin/env python
from __future__ import division
import sys
sys.path.append('../../')

import bisect
import numpy as np
import cv2

from utils.utils import get_roi


# Weight functions

def uniform_weight(N_particles):
    return 1 / N_particles


# Resampling functions

def multinomial_resample(particles, weights, img_boundary):
    cumulative_sum = np.cumsum(weights)
    new_particles = []
    for i in range(len(particles)):
        old_p = particles[bisect.bisect_left(cumulative_sum, np.random.uniform(0, 1))]
        if old_p.w > 0.3:
            new_particles.append([old_p.x, old_p.y, 1/len(particles), old_p.size])
        else:
            rand_x = np.random.randint(img_boundary[0])
            rand_y = np.random.randint(img_boundary[1])
            new_particles.append([rand_x, rand_y, 1/len(particles), old_p.size])
    return new_particles


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

def uniform_displacement(particles, img_boundary, const_dist=10):
    """ Displace particle with const_dist in pixel """
    for p in particles:
        x_old = p.x
        y_old = p.y
        x_new = int(x_old + np.random.uniform(-const_dist, const_dist) + gaussian_noise())
        y_new = int(y_old + np.random.uniform(-const_dist, const_dist) + gaussian_noise())
        p.x = max(0, min(x_new, img_boundary[0]))
        p.y = max(0, min(y_new, img_boundary[1]))
        p.region = ((int(p.x - p.size[0]/2), int(p.y - p.size[1]/2)),
                    (int(p.x + p.size[0]/2), int(p.y + p.size[1]/2)))


# Noise model

def gaussian_noise(sigma=0.2):
    return np.random.normal(0, sigma)


# Likelihood

def predict_mean_hue(particles, img, ref_hue):
    for p in particles:
        roi = get_roi(img, p.region[0], p.region[1])
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mean_hue = np.mean(hsv[..., 0])
        p.w = (255 - abs(mean_hue - 170)) / 255


def predict_color_hist(particles, img, ref_hist):
    for p in particles:
        p.w = compare_hsv_hist(get_roi(img, p.region[0], p.region[1]), ref_hist)


def compare_hsv_hist(img, ref_hist=None):
    """ Calculates likelihood of particle by measuring histogram similarity using
    Bhattacharyya's distance
    """
    p_hist = hsv_histogram(img)
    return 1 - cv2.compareHist(p_hist, ref_hist, cv2.HISTCMP_BHATTACHARYYA)


# Features

def hue_(img, BIN_SIZE=10):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [BIN_SIZE],
                        [0, 180])
    return cv2.normalize(hist, None).flatten()


def hsv_histogram(img, BIN_SIZE=10):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [BIN_SIZE, BIN_SIZE, BIN_SIZE],
                        [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, None).flatten()
