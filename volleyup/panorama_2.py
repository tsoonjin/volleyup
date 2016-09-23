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
        good = [m for m, n in matches if m.distance < ratio * n.distance]
        return good
    
    def calc_translation(self, src_pts, dst_pts):
        m = cv2.estimateRigidTransform(src_pts, dst_pts, True)
        return np.float32([[1, 0, m[0, 2]],
                           [0, 1, 0]])

    def calc_homography(self, imgA, imgB, kp1, kp2, good_matches, min_good_match=4, reproj_thresh=5.0):
        """ Calculates homography and affine transformation when there is at least 8 matched feature points (4 in each image)
        Parameters
        ----------
        min_good_match : minimum number of good matches before calculating homography
        reproj_thresh  : maximum allowed reprojection error for RANSAC to be treated as inlier
        """
        if len(good_matches) >= min_good_match:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
            affine = self.calc_translation(dst_pts, src_pts)
            return (H, status, affine)

        print "Not enough matches are found - %d/%d" % (len(good_matches), min_good_match)
        return None

    def calculate_size(self, imgA_size, imgB_size, homography):
        (h1, w1) = imgA_size[:2]
        (h2, w2) = imgB_size[:2]

        # remap the coordinates of the projected image onto the panorama image space
        top_left = np.dot(homography, np.asarray([0, 0, 1]))
        top_right = np.dot(homography, np.asarray([w2, 0, 1]))
        bottom_left = np.dot(homography, np.asarray([0, h2, 1]))
        bottom_right = np.dot(homography, np.asarray([w2, h2, 1]))

        # normalize
        top_left = top_left/top_left[2]
        top_right = top_right/top_right[2]
        bottom_left = bottom_left/bottom_left[2]
        bottom_right = bottom_right/bottom_right[2]

        pano_left = int(min(top_left[0], bottom_left[0], 0))
        pano_right = int(max(top_right[0], bottom_right[0], w1))
        W = pano_right - pano_left

        pano_top = int(min(top_left[1], top_right[1], 0))
        pano_bottom = int(max(bottom_left[1], bottom_right[1], h1))
        H = pano_bottom - pano_top

        size = (W, H)

        # offset of first image relative to panorama
        X = int(min(top_left[0], bottom_left[0], 0))
        Y = int(min(top_left[1], top_right[1], 0))
        offset = (-X, -Y)
        return (size, offset)

    def overlay_image(self, mainImage, overlayImage):
        return np.where(overlayImage == [0,0,0], mainImage, overlayImage)

    def generate_mosaic(self, mask_func, channel='hsv_s', feature='akaze'):
        """ Generates image mosaics from images
        Parameters
        ----------
        mask_func : mask function to produce mask used to reduce search area of feature detection
        channel   : channel used for processing
        feature   : feature detector for interest point detection

        """
            
        #panorama_img = np.pad(panorama_img, ((0,size[1] - panorama_img.shape[0]),(0,size[0] - panorama_img.shape[1]),(0,0)), mode='constant')
        panorama_img = self.imgs[0]
        panorama_img = np.pad(panorama_img, ((200,200),(200,200),(0,0)), mode='constant')
        for index, imgB in enumerate(self.imgs[1:]):
            key_points_A, desc1 = self.ft.compute(get_channel(panorama_img, channel), feature, panorama_img)
            key_points_B, desc2 = self.ft.compute(get_channel(imgB, channel), feature, imgB)
            matching_features = self.match_features(desc2, desc1)
            (H, status, affine) = self.calc_homography(imgB, panorama_img, key_points_B, key_points_A, matching_features)
            if H is not None:
                (size, offset) = self.calculate_size(imgB.shape, panorama_img.shape, H)
    
                warpedB = cv2.warpPerspective(imgB, H, (panorama_img.shape[1], panorama_img.shape[0]))
                
                warped_key_points_B, warped_desc2 = self.ft.compute(get_channel(warpedB, channel), feature, warpedB)
            
                warped_matching_features = self.match_features(warped_desc2, desc1)
                warped_matched = cv2.drawMatches(warpedB, warped_key_points_B, panorama_img, key_points_A, warped_matching_features, None, flags=2)
                
                panorama_img = self.overlay_image(panorama_img, warpedB)
                #cv2.imshow('matched | warped', warped_matched)
                cv2.imshow('matched | warped', panorama_img)
                cv2.waitKey(0)

        return panorama_img

if __name__ == '__main__':
    # Put extracted images into DATA_DIR before running this
    imgs = get_jpgs(config.DATA_DIR + "beachVolleyball4/")
    cv2.ocl.setUseOpenCL(False)
    stitcher = TranslationStitcher(imgs)
    stitcher.generate_mosaic(get_netmask)
    cv2.destroyAllWindows()
