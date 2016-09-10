#!/usr/bin/env python
""" Feature Detector and Descriptor Class """
import cv2


class FeatureDescriptor():
    """ Feature related computation """
    def __init__(self):
        self.features = {'sift':    cv2.xfeatures2d.SIFT_create(nfeatures=250),
                         'akaze':   cv2.AKAZE_create(),
                         'surf':    cv2.xfeatures2d.SURF_create(600),
                         'orb':     cv2.ORB_create(400),
                         'brisk':   cv2.BRISK_create()}

    def compute(self, img, feature='sift'):
        """ Returns keypoints and descriptors of chosen feature
        Parameters
        ----------
        feature:    feature detector and descriptor used
        """
        return self.features[feature].detectAndCompute(img, None)
