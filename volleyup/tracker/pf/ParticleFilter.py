#!/usr/bin/env python
from __future__ import division
import sys
sys.append('../')

import numpy as np
import cv2

import config

from utils import get_jpgs, draw_str, get_courtmask, get_playermask
from pf_utils import (uniform_weight, systematic_resample, multinomial_resample, uniform_displacement,
                      predict_color_hist, hsv_histogram, predict_mean_hue)


class PlayerParticle():
    """ Represents a player as a state, (x, y, w, h)
        """
    def __init__(self, x, y, w, size):
        self.x = x
        self.y = y
        # Normalized weight that always sum to 1
        self.w = w
        self.size = size
        # Rectangle region represented as (top_left, bottom_right)
        self.region = ((self.x - self.size[0]/2, self.y - self.size[1]/2),
                       (self.x + self.size[0]/2, self.y + self.size[1]/2))

    def draw(self, canvas, region=False):
        """ Draw location of particle on canvas
            region: Draw bounding box if true
        """
        cv2.circle(canvas, (self.x, self.y), 2, config.RED, -1)
        if region:
            cv2.rectangle(canvas, self.region[0], self.region[1], config.RED, 1)

    @staticmethod
    def generate(N_particles, img_boundary, size, weight_generator=None):
        """ Generate initial set of particles
            weight_generator   : callable to generate initial weights of particles
        """
        xs = np.random.randint(img_boundary[0], size=N_particles)
        ys = np.random.randint(img_boundary[1], size=N_particles)
        return [PlayerParticle(x, y, uniform_weight(N_particles), size) for x, y in zip(xs, ys)]


class ParticleFilter():
    """ Generic particle filter for object tracking for single object """
    def __init__(self, particle_generator,
                 img_boundary=(638, 356), N_particles=400, size=(20, 30),
                 ref_img=cv2.imread("{}green.jpg".format(config.TEMPLATE_DIR)),
                 transition_model=uniform_displacement,
                 resampling_handler=multinomial_resample,
                 prediction_model=predict_color_hist
                 ):
        """ Initialize particle filter with given particle generator
            particle generator : callable to generate particles used for tracking
            img_boundary       : dimension of image to be tracked
            size               : region covered by particle
            transition_model   : callable that determines movement of particles
            resampling_handler : callable to resample particles according to weight
        """
        print("N: {}\nBoundary: {}\nSize: {}".format(N_particles, img_boundary, size))
        self.particles = particle_generator(N_particles, img_boundary, size)
        self.ref_img = ref_img
        self.img_boundary = img_boundary
        self.transition = transition_model
        self.resampling_handler = resampling_handler
        self.predict = prediction_model

    def process(self, img):
        # Displacement
        self.transition(self.particles, self.img_boundary)
        # Prediction
        # self.predict(self.particles, img, np.mean(cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2HSV)[..., 0]))
        self.predict(self.particles, img, hsv_histogram(self.ref_img))
        for p in self.particles:
            # draw_str(img, (p.x, p.y), "{0:.2f}".format(p.w))
            p.draw(img)

        # Resampling
        sum_w = sum([p.w for p in self.particles])
        ps = self.resampling_handler(self.particles, [p.w/sum_w for p in self.particles], self.img_boundary)
        self.particles = [PlayerParticle(*p) for p in ps]


if __name__ == '__main__':
    imgs = get_jpgs(config.INDVIDUAL_VIDEOS['1'])
    imgs = get_jpgs(config.INPUT_IMGS)
    pf = ParticleFilter(PlayerParticle.generate, img_boundary=(imgs[0].shape[1], imgs[0].shape[0]))
    for img in imgs:
        pf.process(img)
        cv2.imshow('particle filter', img)
        k = cv2.waitKey(1)
        if k == 27:
            break
