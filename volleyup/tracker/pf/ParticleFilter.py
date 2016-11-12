#!/usr/bin/env python
from __future__ import division
import sys
sys.path.append('../../')

import numpy as np
import cv2

from utils import config

from bg_subtract.bg_subtract import gmm_mog2
from utils.utils import get_jpgs, draw_str, get_courtmask, get_playermask
from pf_utils import (uniform_weight, systematic_resample, multinomial_resample, uniform_displacement,
                      predict_color_hist, hsv_histogram, predict_mean_hue, normalize_particles)


class PlayerParticle():
    """ Represents a player as a state, (x, y, w, h)
        """
    def __init__(self, x, y, w, size, color=config.RED):
        self.x = x
        self.y = y
        # Normalized weight that always sum to 1
        self.w = w
        self.size = size
        # Rectangle region represented as (top_left, bottom_right)
        self.region = ((self.x - self.size[0]/2, self.y - self.size[1]),
                       (self.x + self.size[0]/2, self.y))
        self.color = color

    def draw(self, canvas, region=False):
        """ Draw location of particle on canvas
            region: Draw bounding box if true
        """
        if region:
            cv2.rectangle(canvas, self.region[0], self.region[1], self.color, 2)
        cv2.circle(canvas, (self.x, self.y), 3, config.PURPLE, -1)

    @staticmethod
    def generate(N_particles, img_boundary, size, color, weight_generator=None):
        """ Generate initial set of particles
            weight_generator   : callable to generate initial weights of particles
        """
        border_offset = (int(size[0] / 2), int(size[1] / 2))
        xs = np.random.randint(border_offset[0], img_boundary[0] - border_offset[0], size=N_particles)
        ys = np.random.randint(border_offset[1], img_boundary[1] - border_offset[1], size=N_particles)
        return [PlayerParticle(x, y, uniform_weight(N_particles), size, color) for x, y in zip(xs, ys)]


class ParticleFilter():
    """ Generic particle filter for object tracking for single object """
    def __init__(self, particle_generator,
                 img_boundary, N_particles=1000, size=(30, 60),
                 ref_img=cv2.imread("{}lat_green_far.png".format(config.TEMPLATE_DIR)),
                 color=config.RED,
                 background_model=gmm_mog2(),
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
        print("N: {}\nBoundary: {}\nSize: {}".format(N_particles,
                                                     (img_boundary[0], img_boundary[1]),
                                                     size))
        self.tracked = None
        self.particles = particle_generator(N_particles, img_boundary, size, color)
        self.ref_img = ref_img
        self.img_boundary = img_boundary
        self.transition = transition_model
        self.resampling_handler = resampling_handler
        self.predict = prediction_model

    def process(self, img, detected=None):
        # Displacement
        self.transition(self.particles, self.img_boundary)
        # Prediction
        # self.predict(self.particles, img, np.mean(cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2HSV)[..., 0]))
        self.predict(self.particles, img, hsv_histogram(self.ref_img))
        self.particles = sorted(self.particles, key=lambda x: x.w, reverse=True)
        max_w = max([p.w for p in self.particles])
        high_ps = [p for p in self.particles if p.w > 0.8*max_w]
        high_ps = sorted(high_ps, key=lambda x: x.w, reverse=True)
        center = (-1, -1)
        if high_ps[0].w > 0.4:
            center = self.weighted_center(high_ps[:20])
            region = self.calc_region(center[0], center[1], self.particles[0].size)
            cv2.rectangle(img, region[0], region[1], self.particles[0].color, 2)
            cv2.circle(img, (center[0], center[1]), 3, config.PURPLE, -1)
        for p in high_ps[:20]:
            # p.draw(img, True)
            pass

        # Resampling
        sum_w = sum([p.w for p in self.particles])
        ps = self.resampling_handler(self.particles, [p.w/sum_w for p in self.particles], self.img_boundary)
        self.particles = [PlayerParticle(*p) for p in ps]
        return img, center

    def calc_region(self, x, y, size):
        return ((int(x - size[0]/2), int(y - size[1])), (int(x + size[0]/2), y))

    def weighted_center(self, particles):
        sum_w = sum([p.w for p in particles])
        weights = [p.w / sum_w for p in particles]
        x = sum([w*p.x for w, p in zip(weights, particles)])
        y = sum([w*p.y for w, p in zip(weights, particles)])
        return (int(x), int(y))


def generate_player_config(vid_id, ref_img_path, color=config.PURPLE, size=(30, 60)):
    img_boundary = config.IMG_BOUNDARY[vid_id]
    ref_img = cv2.imread('{}{}'.format(config.TEMPLATE_DIR, ref_img_path))
    return ParticleFilter(PlayerParticle.generate, img_boundary=img_boundary,
                          ref_img=ref_img, color=color, size=size)


''' Video specific configurations '''
VID_PFS = {'1': [generate_player_config(1, 'bra_white_far1_1.png', config.BRAZIL['color'], size=(30, 40)),
                 generate_player_config(1, 'bra_white_far1_2.png', config.BRAZIL['color'], size=(30, 40)),
                 generate_player_config(1, 'lat_green_far1.png', config.LATVIA['color']),
                 generate_player_config(1, 'lat_green_near1.png', config.LATVIA['color'])],
           '2': [generate_player_config(2, 'bra_white_far2_1.png', config.BRAZIL['color'], size=(30, 40)),
                 generate_player_config(2, 'bra_white_far2.png', config.BRAZIL['color'], size=(30, 40)),
                 generate_player_config(2, 'lat_green_far2.png', config.LATVIA['color']),
                 generate_player_config(2, 'lat_green_near2.png', config.LATVIA['color'])],
           '3': [generate_player_config(3, 'bra_white_far3.png', config.BRAZIL['color'], size=(30, 40)),
                 generate_player_config(3, 'bra_white_near3.png', config.BRAZIL['color'], size=(30, 40)),
                 generate_player_config(3, 'lat_green_far3.png', config.LATVIA['color'], size=(30, 40)),
                 generate_player_config(3, 'lat_green_near3.png', config.LATVIA['color'], size=(30, 40))],
           '4': [generate_player_config(4, 'bra_white_far4.png', config.BRAZIL['color'], size=(30, 40)),
                 generate_player_config(4, 'bra_white_near4.png', config.BRAZIL['color'], size=(30, 40)),
                 generate_player_config(4, 'lat_green_far4.png', config.LATVIA['color'], size=(30, 40)),
                 generate_player_config(4, 'lat_green_near4.png', config.LATVIA['color'], size=(30, 40))],
           '5': [generate_player_config(5, 'esp_white_far5_1.png', config.ESPANYOL['color'], size=(20, 30)),
                 generate_player_config(5, 'esp_white_far5_2.png', config.ESPANYOL['color'], size=(20, 30)),
                 generate_player_config(5, 'usa_red_far5.png', config.USA['color'], size=(25, 45)),
                 generate_player_config(5, 'usa_red_near5.png', config.USA['color'], size=(30, 60))],
           '6': [generate_player_config(6, 'esp_white_near6.png', config.ESPANYOL['color'], size=(30, 40)),
                 generate_player_config(6, 'esp_white_far6.png', config.ESPANYOL['color'], size=(30, 40)),
                 generate_player_config(6, 'usa_red_far6.png', config.USA['color'], size=(30, 40)),
                 generate_player_config(6, 'usa_red_near6.png', config.USA['color'], size=(30, 40))],
           '7': [generate_player_config(7, 'esp_white_near7.png', config.ESPANYOL['color'], size=(30, 40)),
                 generate_player_config(7, 'esp_white_far7.png', config.ESPANYOL['color'], size=(30, 40)),
                 generate_player_config(7, 'usa_red_far7.png', config.USA['color'], size=(30, 40)),
                 generate_player_config(7, 'usa_red_near7.png', config.USA['color'], size=(30, 40))]}


if __name__ == '__main__':
    pass
