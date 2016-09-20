#!/usr/bin/env python
""" Performs image stitching on images provided """
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

    def calc_homography_affine(self, imgA, imgB, kp1, kp2, good_matches,
                               min_good_match=4, reproj_thresh=4.0):
        """ Calculates homography and affine transformation when there is at least 8 matches
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

    def merge_images_translation(self, imgA, imgB, offset):
        # Put images side-by-side into 'image'.
        (h1, w1) = imgA.shape[:2]
        (h2, w2) = imgB.shape[:2]
        (ox, oy) = offset
        ox = int(ox)
        oy = int(oy)
        h, w = max(h1, h2), max(w1, w2)
        image = np.zeros((h, w + abs(ox), 3), np.uint8)
        image[:h1, :w1] = imgA
        image[:h2-oy, ox:ox+w2] = imgB[oy:h2, :]
        return image

    def generate_mosaic(self, mask_func, channel='hsv_s', feature='akaze'):
        """ Generates image mosaics from images
        Parameters
        ----------
        mask_func : mask function to produce mask used to reduce search area of feature detection
        channel   : channel used for processing
        feature   : feature detector for interest point detection

        """
        panorama_img = self.imgs[0]
        prev_img = panorama_img.copy()
        for index, next_img in enumerate(self.imgs[1:]):
            kp1, desc1 = self.ft.compute(get_channel(panorama_img, channel), feature,
                                         mask_func(panorama_img))
            kp2, desc2 = self.ft.compute(get_channel(next_img, channel), feature,
                                         mask_func(next_img))
            matches = self.match_features(desc1, desc2)
            (H, status, affine) = self.calc_homography_affine(panorama_img, next_img, kp1,
                                                              kp2, matches)
            if H is not None:
                (size, offset) = self.calculate_size(panorama_img.shape, panorama_img.shape, H)
                self.debug_matching(prev_img, next_img, mask_func, channel, feature)
            panorama_img = next_img.copy()
            prev_img = next_img.copy()
        return panorama_img

    def get_avg_translation(self, kp1, kp2, matches):
        mean_x = np.mean([kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0] for m in matches])
        mean_y = np.mean([kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1] for m in matches])
        return int(mean_x), int(mean_y)

    def debug_matching(self, imgA, imgB, mask_func, channel, feature, wait=0):
        """ Displays debugging window for each images along with detected keypoints """
        kp1, desc1 = self.ft.compute(get_channel(imgA, channel), feature,
                                     mask_func(imgA))
        kp2, desc2 = self.ft.compute(get_channel(imgB, channel), feature,
                                     mask_func(imgB))
        matches = self.match_features(desc1, desc2)
        (H, status, affine) = self.calc_homography_affine(imgA, imgB, kp1, kp2, matches)
        cv2.drawKeypoints(imgA, kp1, imgA, config.BLUE, 1)
        cv2.drawKeypoints(imgB, kp2, imgB, config.BLUE, 1)
        warpedA = cv2.warpPerspective(imgA, H, (imgB.shape[1], imgB.shape[0]))
        warpedB = cv2.warpPerspective(imgB, np.linalg.inv(H), (imgA.shape[1], imgA.shape[0]))
        matched = cv2.drawMatches(imgA, kp1, imgB, kp2, matches, None, flags=2)
        cv2.imshow('matched | warped', np.vstack((matched, np.hstack((warpedA, warpedB)))))
        cv2.waitKey(wait)


if __name__ == '__main__':
    imgs = get_jpgs(config.INDVIDUAL_VIDEOS['3'], skip=3)
    stitcher = TranslationStitcher(imgs)
    stitcher.generate_mosaic(get_netmask)
    cv2.destroyAllWindows()
