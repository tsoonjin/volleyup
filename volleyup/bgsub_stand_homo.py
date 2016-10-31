import cv2
import numpy as np
from histogram import CourtFinder

class LK_homo_bg:
    """ Optical Flow tracking using Lucas-Kanade method based on sample/python/lk_track.py """
    def __init__(self):
        # Config
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.frame_idx = 0

    def track_frame(self, frame, mask):
        """
        takes a frame and track
        """
        homography = np.float32(np.matrix(np.identity(3)))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if len(self.tracks) > 0:
            img0, img1 = self.prev_gray, frame_gray

            #
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_tracks = []
            homo_points_0 = []
            homo_points_1 = []
            for tr, (x0, y0), (x, y), good_flag in zip(self.tracks, p0.reshape(-1, 2), p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                if len(tr) > self.track_len:
                    del tr[0]
                new_tracks.append(tr)
                homo_points_0.append((x0, y0))
                homo_points_1.append((x, y))
            self.tracks = new_tracks
            homography = cv2.findHomography(np.int32(homo_points_0), np.int32(homo_points_1), cv2.RANSAC, 5)[0]

        if self.frame_idx % self.detect_interval == 0:
            p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    self.tracks.append([(x, y)])

        self.frame_idx += 1
        self.prev_gray = frame_gray

        return (np.int32([tr[-1] for tr in self.tracks]).reshape(-1, 2), homography)


        




video = cv2.VideoCapture("data/beachVolleyball1.mov")
ret, oframe = video.read()
cf = CourtFinder()
tracker = LK_homo_bg()
acounter = 0

background = np.float32(oframe)
background_gray = np.float32(cv2.cvtColor(oframe, cv2.COLOR_BGR2GRAY))
backgrounds_gray = []

while True:
    acounter += 1
    ret, frame = video.read()
    cols, rows, *_ = frame.shape
    if not ret:
        break
    mask =  255 - cf.process_frame(frame)



    features = frame.copy()
    pts, homography = tracker.track_frame(frame, mask)
    # print(pts)
    for (x, y) in pts:
        cv2.circle(features, (x,y), 5, 0, -1)

    print([homography])
    background = cv2.warpPerspective(background, homography, (rows, cols) )
    alpha = 1/(acounter +1)
    subframe = np.float32(frame) *(1-alpha)
    print(background.shape)
    print(subframe.shape)
    cv2.accumulateWeighted(subframe, background, alpha)
    bg = cv2.convertScaleAbs(background)

    print("GRAY")
    background_gray = cv2.warpPerspective(background_gray, homography, (rows, cols) )
    alpha = 1/(acounter +1)
    subframe = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) *(1-alpha)
    print(background_gray.shape)
    print(subframe.shape)
    print([subframe])
    cv2.accumulateWeighted(subframe, background_gray, alpha)
    bg_gray = cv2.convertScaleAbs(background_gray)
    backgrounds_gray.append(bg_gray)
    if acounter > 3:
        for i in range(-3, 0):
            backgrounds_gray[i] = cv2.warpPerspective(background_gray, homography, (rows, cols))
        print(backgrounds_gray[-3:])
        bgmed_gray = np.median(backgrounds_gray[-3:], axis = 0)
        bgmean_gray = np.mean(backgrounds_gray[-3:], axis = 0)


    else:
        bgmed_gray = bg_gray
        bgmean_gray = bg_gray



    cv2.imshow("mask", mask)
    cv2.imshow("features", features)
    cv2.imshow("background", bg)
    cv2.imshow("subtracted", frame - bg)
    cv2.imshow("background_gray", bg_gray)
    print("bgmed")
    print(bgmed_gray)
    cv2.imshow("background_gray_median", bgmed_gray)
    cv2.imshow("subtracted", cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) - bg_gray)
    cv2.imshow("subtracted_med", cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) - bgmed_gray)
    print(homography)










    k = cv2.waitKey(10)
    if k == 27:
        print("Terminated by user")
        exit()


