#!/usr/bin/env python
""" Performs image stitching on images provided, to create a panorama view """
import cv2
import numpy as np
import math
import config
from utils import get_channel, get_jpgs, get_netmask
from feature import FeatureDescriptor


class TranslationStitcher():
    """ Assumes only translation offset between images and credited to
        https://github.com/marcpare/stitch/blob/master/crichardt/stitch.py
    """
    def __init__(self, imgs):
        self.ft = FeatureDescriptor()
        self.imgs = imgs

    def calc_matches(self, desc1, desc2, method='bf'):
        """ Calculate matches between descriptors specified by given method
        Parameters
        ----------
        method : bf    (brute force matching)
                 flann (fast nearest neighbor matching)

        """
        if method is 'bf':
            bf = cv2.BFMatcher()
            return bf.knnMatch(desc1, desc2, k=2)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        return flann.knnMatch(desc1, desc2, k=2)

    def match_features(self, desc1, desc2, ratio=0.8):
        """ Matches features and filter only good matches using Lowe's ratio """
        matches = self.calc_matches(desc1, desc2)
        good_matches = [m for m, n in matches if m.distance < ratio * n.distance]
        return good_matches
    
    def calc_translation(self, src_pts, dst_pts):
        m = cv2.estimateRigidTransform(src_pts, dst_pts, True)
        return np.float32([[1, 0, m[0, 2]],
                           [0, 1, 0]])

    def calc_homography(self, imgA, imgB, kp1, kp2, good_matches, min_good_match=4, reproj_thresh=2.0):
        """ 
        Calculates homography when there are at least 8 matched feature points (4 in each image)
        Parameters
        ----------
        min_good_match : minimum number of good matches before calculating homography
        reproj_thresh  : maximum allowed reprojection error for RANSAC to be treated as inlier
        """
        if len(good_matches) >= min_good_match:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
            return (H, status)

        print "Not enough matches are found - %d/%d" % (len(good_matches), min_good_match)
        return None

    def calc_affine(self, src_pts, dst_pts):
        """
            Calculates affine transformation required for destination points of the destination image to match the source points
        """
        affine = self.calc_translation(dst_pts, src_pts)
        return affine

    def overlay_image(self, mainImage, overlayImage):
        """
           Overlays overlayImage onto mainImage, assuming that important parts of overlayImage are not totally black i.e. (0,0,0)
        """
        return np.where(overlayImage == [0,0,0], mainImage, overlayImage)

    def generate_panorama(self, mask_func, channel='hsv_s', feature='akaze'):
        """ Generates image panorama
        Parameters
        ----------
        mask_func : mask function to produce mask used to reduce search area of feature detection (currently, this is not used)
        channel   : channel used for processing
        feature   : feature detector for interest point detection

        """
        panorama_img = self.imgs[0]
        # Add in margins in all directions
        panorama_img = np.pad(panorama_img, ((200,200),(200,200),(0,0)), mode='constant')
        for index, imgB in enumerate(self.imgs[1:]):
            # Find key features in each of the grayscale images, seems to perform better
            key_points_A, desc1 = self.ft.compute(get_channel(panorama_img, channel), feature, cv2.cvtColor(panorama_img, cv2.COLOR_RGB2GRAY))
            key_points_B, desc2 = self.ft.compute(get_channel(imgB, channel), feature, cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY))
            # Match feature descriptors and filter which keeps the good ones
            matching_features = self.match_features(desc2, desc1)
            # Calculate the homography matrix and affine required to transform imgB to panorama_img (so that the matching points overlap)
            (H, status) = self.calc_homography(imgB, panorama_img, key_points_B, key_points_A, matching_features)
            if H is not None:
                warpedB = cv2.warpPerspective(imgB, H, (panorama_img.shape[1], panorama_img.shape[0]))
                panorama_img = self.overlay_image(panorama_img, warpedB)
                cv2.imshow('panorama', panorama_img)
                cv2.waitKey(0)

        return panorama_img

if __name__ == '__main__':
    # Put extracted images into DATA_DIR/<folder> before running this
    imgs = get_jpgs(config.DATA_DIR + "beachVolleyball1/")
    cv2.ocl.setUseOpenCL(False) # A workaround for ORB feature detector error
    stitcher = TranslationStitcher(imgs)
    stitcher.generate_panorama(get_netmask)
    cv2.destroyAllWindows()
