#!/usr/bin/env python
""" Performs image stitching on images provided """
import cv2
import numpy as np
from feature import FeatureDescriptor


class TranslationStitcher():
    """ Assumes only translation offset between images and credited to
        https://github.com/marcpare/stitch/blob/master/crichardt/stitch.py
    """
    def __init__(self, imgs):
        self.ft = FeatureDescriptor()
        self.imgs = imgs

    def match_features(self, desc1, desc2, ratio=0.5):
        """ Matches features using FlannBasedMatcher with Lowe's ratio given """
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)
        good = [m for m, n in matches if m.distance < ratio * n.distance]
        return good

    def calc_homography(self, imgA, imgB, kp1, kp2, good_matches,
                        min_good_match=8, reproj_thresh=5.0):
        """ Calculates homography when there is at least 4 matches
        Parameters
        ----------
        min_good_match : minimum number of good matches before calculating homography
        reproj_thresh  : maximum allowed reprojection error for RANSAC to be treated as inlier
        """
        if len(good_matches) > min_good_match:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
            return (H, status)

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
        image = np.zeros((h, w*2, 3), np.uint8)
        image[:h1, :w1] = imgA
        image[:h2-oy, ox:ox+w2] = imgB[oy:h2, :]
        return image

    def generate_mosaic(self):
        base_img = self.imgs[0]
        for next_img in self.imgs[1:]:
            kp1, desc1 = self.ft.compute(base_img, 'surf')
            kp2, desc2 = self.ft.compute(next_img, 'surf')
            (H, status) = self.calc_homography(base_img, next_img, kp1, kp2,
                                               self.match_features(desc1, desc2))
            if H is not None:
                (size, offset) = self.calculate_size(base_img.shape, base_img.shape, H)
                base_img = self.merge_images_translation(base_img, next_img, offset)
                cv2.imshow('panorama', base_img)
                cv2.waitKey(10)
        return base_img


if __name__ == '__main__':
    imgs = [cv2.imread('data/stub/m{}.png'.format(i)) for i in range(1, 5)]
    stitcher = TranslationStitcher(imgs)
    cv2.imshow('panorama', stitcher.generate_mosaic())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
