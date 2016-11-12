import cv2
import numpy as np
from histogram import CourtFinder
from skimage.feature import hog
from sklearn.externals import joblib
import os

def subtract(original, deductible):
    return np.uint8((original - np.int32(deductible)).clip(min = 0))

def er_filter(image):
    kernel = np.ones((3,3), np.uint8)
    image = cv2.erode(image, kernel, iterations = 1)
    image = cv2.dilate(image, kernel, iterations = 1)
    return image

def gaussian_noise_filter(image):
    blur = cv2.GaussianBlur(image, (5,5), 0 )
    thresh = cv2.threshold(blur, 33, 255, cv2.THRESH_BINARY)
    return thresh[1]


class LK_homo_tracker:
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

class HomoBG:
    """
    Background subtractor
    Uses median and homology to perform background subtraction
    window_size: integer. Size of window to keep track of for background model calculation
    """
    def __init__(self, window_size = 5):
        self.tracker = LK_homo_tracker()
        self.court_finder = CourtFinder()
        self.window_size =5
        self.init_frame()

    def run_test(self, video_path):
        video = cv2.VideoCapture(video_path)
        backgrounds_gray = []
        acounter = 0

        while True:
            acounter += 1
            ret, frame = video.read()
            if not ret:
                break
            cols, rows, *_ = frame.shape
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # bgmed_gray = np.zeros((rows, cols))

            mask =  255 - self.court_finder.process_frame(frame)
            pts, homography = self.tracker.track_frame(frame, mask)
            if acounter > self.window_size:
                for i in range(-self.window_size, 0):
                    backgrounds_gray[i] = cv2.warpPerspective(backgrounds_gray[i], homography, (rows, cols))
                bgmed_gray = np.median(backgrounds_gray[-self.window_size:], axis = 0)
            backgrounds_gray.append(frame_gray)

            cv2.imshow("original", frame)
            if acounter > self.window_size:
                cv2.imshow("subtracted_-2", subtract(frame_gray, backgrounds_gray[-2]))
                cv2.imshow("subtracted_med", subtract(frame_gray, bgmed_gray))
            k = cv2.waitKey(10)
            if k == 27:
                print("Terminated by user")
                exit()

    def run(self, video_path, filter_funct = lambda x: x, startfrom = 0):
        video = cv2.VideoCapture(video_path)
        backgrounds_gray = []
        acounter = 0

        while True:
            acounter += 1
            
            ret, frame = video.read()
            if acounter < startfrom:
                continue
            if not ret:
                break
            cols, rows, *_ = frame.shape
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # bgmed_gray = np.zeros((rows, cols))

            mask =  255 - self.court_finder.process_frame(frame)
            pts, homography = self.tracker.track_frame(frame, mask)
            if acounter - startfrom > self.window_size:
                for i in range(-self.window_size, 0):
                    backgrounds_gray[i] = cv2.warpPerspective(backgrounds_gray[i], homography, (rows, cols))
                bgmed_gray = np.median(backgrounds_gray[-self.window_size:], axis = 0)
            backgrounds_gray.append(frame_gray)

            if acounter - startfrom > self.window_size:
                yield (filter_funct(subtract(frame_gray, bgmed_gray)), frame_gray, frame)

    def run_frames(self, frames, filter_funct = lambda x: x, startfrom = 0):
        backgrounds_gray = []
        for counter, frame in enumerate(frames):
            cols, rows, *_ = frame.shape
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = 255 - self.court_finder.process_frame(frame)
            pts, homography = self.tracker.track_frame(frame, mask)
            if counter - startfrom > self.window_size:
                for i in range(-self.window_size, 0):
                    backgrounds_gray[i] = cv2.warpPerspective(backgrounds_gray[i], homography, (rows, cols))
                bgmed_gray = np.median(backgrounds_gray[-self.window_size:], axis = 0)
            backgrounds_gray.append(frame_gray)
            if counter - startfrom > self.window_size:
                yield (filter_funct(subtract(frame_gray, bgmed_gray)), frame_gray, frame)

    def init_frame(self):
        self.backgrounds_gray = []
        self.counter = 0

    def run_frame(self, frame, filter_funct = lambda x: x):
        self.counter += 1
        cols, rows, *_ = frame.shape
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = 255 - self.court_finder.process_frame(frame)
        pts, homography = self.tracker.track_frame(frame, mask)
        if self.counter > self.window_size:
            for i in range(-self.window_size, 0):
                self.backgrounds_gray[i] = cv2.warpPerspective(self.backgrounds_gray[i], homography, (rows, cols))
            bgmed_gray = np.median(self.backgrounds_gray[-self.window_size:], axis = 0)
            cv2.imshow("bgmed_gray", np.uint8(bgmed_gray))
        self.backgrounds_gray.append(frame_gray)
        if self.counter > self.window_size:
            return (filter_funct(subtract(frame_gray, bgmed_gray)), frame_gray, frame)
        return None

class BallFinder:
    def __init__(self, filter_funct = er_filter, window_size = 5):
        self.filter_funct = filter_funct
        self.bgsub = HomoBG(window_size = 5)
        self.init_frame()
    
    def init_frame(self):
        self.bgsub.init_frame()

    def run_frame(self, frame):
        res = self.bgsub.run_frame(frame, filter_funct = self.filter_funct)
        if not res:
            return None
        subbed, gray, frame = res

        subbed[0:5, :] = 0
        subbed[:, 628:632] = 0
        subbed[:, 0:7] = 0
        subbed[295:300, :] = 0
        subbed[250:300, 532:632] = 0
        cv2.imshow("subbed", subbed)

        circles = cv2.HoughCircles(subbed, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2 =10, minRadius =1, maxRadius = 7)
        if circles is not None:
            print(circles)
            return circles[0]
        else:
            print(circles)
            return circles




        

if __name__ == "__main__":
    a = HomoBG(window_size = 3)
    #a.run_test()
    startfrom = 0
    video_path = "data/beachVolleyball4.mov"
    output_dir = "output/tests/bv3"
    tt_dir = "data/training_data/bgsub/bv4"
    svm = joblib.load("data/svm/SVM_HN_second.pkl")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(tt_dir):
        os.makedirs(tt_dir)

    video = cv2.VideoCapture(video_path) 

    for i in range(startfrom):
        ret, frame = video.read()
    f_video = cv2.VideoCapture(video_path)

    #Initialize frames for testing
    frames = []
    while True:
        ret, frame = f_video.read()
        if not ret or frame == None:
            break
        frames.append(frame)

    # bf = BallFinder()
    
    # for frame in frames:
    #     circles = bf.run_frame(frame)
    #     if circles is not None:
    #         for (x,y,r) in circles:
    #             cv2.circle(frame, (x,y), r, (0,0,255), 2)

    #     cv2.imshow("draw", frame)
    #     k = cv2.waitKey(10)
    #     if k == 27:
    #         print("Terminated by user")
    #         exit()



    frame

    for num, (i, gframe, frame) in enumerate(a.run_frames(frames, filter_funct = er_filter, startfrom = startfrom)):
    # mov
    # for num, (i, gframe, frame) in enumerate(a.run(video_path, filter_funct = er_filter, startfrom = startfrom)):
        oframe = frame.copy()
        #handle the edges because of homology bs
        i[0:5, :] = 0
        i[:, 628:632] = 0
        i[:, 0:7] = 0
        i[295:300, :] = 0


        frame[0:5, :] = 0
        frame[:, 628:632] = 0
        frame[:, 0:7] = 0
        frame[295:300, :] = 0

        i[250:300, 532:632] = 0
        frame[250:300, 532:632] = 0

        cv2.imshow("test", i)
        circles = cv2.HoughCircles(i, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2 =10, minRadius =1, maxRadius = 7)
        if circles is not None:
            print(circles)
            for (x, y, r) in circles[0]:
                target_image = cv2.cvtColor(frame[y-25:y+25, x-25:x+25], cv2.COLOR_BGR2GRAY)
                if target_image is None or target_image.shape != (50, 50):
                    continue
                if svm.predict(hog(target_image).reshape(1, -1))[0]:
                    cv2.circle(frame, (x,y), r, (0, 0, 255), 2)
#                cv2.circle(frame, (x,y), r, (0, 0, 255), 2)
        cv2.imshow("draw", frame)
        #Write outputs
        # cv2.imwrite(os.path.join(output_dir, "{}.png".format(str(num).zfill(4))), frame)

        #Write train/test data
        try:
            if circles is not None:
                print(circles)
                for (x, y, r) in circles[0]:
                    image = oframe[y-50:y+50, x-50:x+50]
                    image2 = oframe.copy()
                    cv2.circle(image2, (x, y), 4, (0, 0, 255), 3)
                    image2 = image2[y-50:y+50, x-50:x+50]
                    # cv2.imshow("image", image2)
                    # cv2.imwrite(os.path.join(tt_dir, "{}.png".format(str(num).zfill(4))), image)
                    # cv2.imwrite(os.path.join(tt_dir, "{}_drawn.png".format(str(num).zfill(4))), image2)

        except Exception as err:
            print(err, x, y, frame.shape)




        k = cv2.waitKey(10)
        if k == 27:
            print("Terminated by user")
            exit()
