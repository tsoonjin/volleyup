#!/usr/bin/env python
import cv2

import config

class FeatureDescriptor():
    """ Feature related computation """
    def __init__(self):
        self.features = {'sift':    cv2.xfeatures2d.SIFT_create(nfeatures=250),
                         'akaze':   cv2.AKAZE_create(),
                         'surf':    cv2.xfeatures2d.SURF_create(600),
                         'orb':     cv2.ORB_create(400),
                         'brisk':   cv2.BRISK_create(),
                         'hog':     cv2.HOGDescriptor()}
        self.imgs = []

    def compute(self, img, feature='sift', mask=None):
        """ Returns keypoints and descriptors of chosen feature
        Parameters
        ----------
        feature:    feature detector and descriptor used

        Returns
        -------
        (keypoints, descriptors)
        """
        return self.features[feature].detectAndCompute(img, mask)

    def draw_features(self, canvas, keypoints, color=config.RED, rad=1):
        cv2.drawKeypoints(canvas, keypoints, canvas, color, rad)
