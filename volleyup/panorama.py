#!/usr/bin/env python
""" Performs image stitching on images provided """
import cv2
import numpy as np

import config


def cv_stitcher():
    """ Returns OpenCV stitcher class for stitching """
    stitcher = cv2.createStitcher(False)
    return stitcher


def detect_features(img, feature="SIFT"):
    """ Detect and return feature descriptors given an RGB image """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Extract from single channel only to save time
    # Detection of features from image
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(gray, None)
    draw_kps(img, kps)
    return (np.float32([kp.pt for kp in kps]), features)


def match_kps(kpsA, kpsB, featuresA, featuresB, ratio, reproj_thresh):
    matcher = cv2.BFMatcher()
    rawMatches = matcher.knnMatch(featuresA, featuresB, k=2)
    matches = [(m[0].trainIdx, m[0].queryIdx) for m in rawMatches if is_accurate_match(m, ratio)]
    return calc_homography(kpsA, kpsB, matches, reproj_thresh)


def calc_homography(kpsA, kpsB, matches, reproj_thresh):
    """ Calculates homography when there is at least 4 matches
    Parameters
    ----------
    reproj_thresh : float
        maximum allowed reprojection error for RANSAC to be treated as inlier
    """
    if len(matches) > 4:
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reproj_thresh)
        return (matches, H, status)
    return None


def is_accurate_match(match, ratio):
    """ Uses Lowe's ratio test to remove false positive match """
    return len(match) == 2 and match[0].distance < match[1].distance * ratio


def stitch(imgA, imgB, ratio=0.75, reproj_thresh=4.0):
    """ Stitches two images together via feature matching
    Parameters
    ----------
    ratio : float
        maximum distance between pair of images to remove false positive
    reproj_thresh : float
        maximum allowed reprojection error for RANSAC to be treated as inlier
    """
    kpsA, featuresA = detect_features(imgA)
    kpsB, featuresB = detect_features(imgB)
    M = match_kps(kpsA, kpsB, featuresA, featuresB, ratio, reproj_thresh)

    if M is None:
        return M

    (matches, H, status) = M
    result = cv2.warpPerspective(imgA, H, (imgA.shape[1] + imgB.shape[1], imgA.shape[0]))
    # result[0:imgB.shape[0], 0:imgB.shape[1]] = imgB
    return result


def draw_kps(img, kps):
    cv2.drawKeypoints(img, kps, img, color=(255, 0, 0))


if __name__ == '__main__':
    pass
