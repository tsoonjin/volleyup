import cv2
import numpy as np
import os
import bt_util

import random
import itertools
from random import randint

def get_rnd_neighbour(x, y, height, width):
    cx = [i for i in [x + 1, x, x - 1] if (i >= 0 and i < width)]
    cy = [i for i in [y + 1, y, y - 1] if (i >= 0 and i < height)]
    choices = [[cx,cy] for cx, cy in itertools.product(cx, cy) if cx != x or cy != y ]
    choice = (random.sample(choices, 1))[0]
    # print(choice)
    return choice


class VibeBG:
    def __init__(self, nbsamples = 20, req_matches = 2, d_thresh = 20, ssample = 16 ):
        self.nbsamples = nbsamples
        self.req_matches = req_matches
        self.d_thresh = d_thresh
        self.ssample = ssample
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.st_params = dict( maxCorners = 100,
                                qualityLevel = 0.3,
                                minDistance = 7,
                                blockSize = 7 )

    def initialize_bg_pixel(self, x, y, image):
        height, width, *_ = image.shape
        for i in range(2, self.nbsamples):
            nx, ny  = get_rnd_neighbour(x, y, height, width)
            self.samples[i][y][x] = image[ny][nx]
        self.samples[0][y][x] = image[y][x]
        self.samples[1][y][x] = image[y][x]

    def initialize_bg(self, image):
        """
        Initializes background from the first frame
        """
        height, width, *_ = image.shape
        self.samples = [np.zeros_like(image) for i in range(self.nbsamples)]
        for x in range(width):
            for y in range(height):
                self.initialize_bg_pixel(x,y, image)    

    def vibe(self, image):
        """
        derived from vibe algorithm
        http://www.telecom.ulg.ac.be/publi/publications/barnich/Barnich2011ViBe/index.html
        """
        height, width, *_ = image.shape
        self.segmentation_map = np.zeros_like(image)
        for x in range(width):
            for y in range(height):
                # BGFG classification
                count = 0
                for index in range(self.nbsamples):
                    if count >= self.req_matches:
                        break
                    distance = abs(int(image[y][x]) - int(self.samples[index][y][x]))
                    if distance < self.d_thresh:
                        count += 1

                # if pixel is background
                if count >= self.req_matches:
                    # set pixel to background in segmentation map
                    self.segmentation_map[y][x] = 0
                    # Updating of BG model
                    rint = randint(0, self.ssample-1)
                    # Update current pixel model stochiastically
                    if rint == 0:
                        rint = randint(0, self.nbsamples-1)
                        self.samples[rint][y][x] = image[y][x]
                    # Diffuse into neighbouring pixel model stochiastically
                    rint = randint(0, self.ssample-1)
                    if rint == 0:
                        nx, ny = get_rnd_neighbour(x, y, height, width)
                        rint = randint(0, self.nbsamples-1)
                        self.samples[rint][ny][nx] = image[y][x]
                # if pixel is foreground:
                else:
                    self.segmentation_map[y][x] = 255
        print("NONZERO: ", np.count_nonzero(self.segmentation_map))

    def pipeline_frames(self, frames, outdir = "data/testing"):
        frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        print("Initializing background")
        self.initialize_bg(frames[0])
        for counter, frame in enumerate(frames[1:]):
            print("Processing frame ", counter + 1)
            self.vibe(frame)
            cv2.imwrite(os.path.join(outdir, str(counter + 1) + ".png"), self.segmentation_map)

            #blurred and threshed
            blur = cv2.GaussianBlur(self.segmentation_map, (5,5), 0)
            cv2.imshow("blur", blur)
            _, thresh = cv2.threshold(blur, 253, 255, cv2.THRESH_OTSU)

            #eroded and dialated
            er = cv2.morphologyEx(self.segmentation_map, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

            cv2.imwrite(outdir + "/gf" + str(counter + 1) + ".png", thresh)
            cv2.imwrite(outdir + "/er" + str(counter + 1) + ".png", er)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

    def pipeline(self, video, outdir = "data/testing"):
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        ret, frame = video.read()
        print("initializing background")
        self.initialize_bg(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        print("Processing video")
        counter = 0
        while True:
            counter += 1
            print("Processing frame ", counter)
            ret, frame = video.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print("vibing")
            self.vibe(gray)
            cv2.imshow("segmap", self.segmentation_map)
            cv2.imshow("original", frame)
            cv2.imwrite(os.path.join(outdir, str(counter) + ".png"), self.segmentation_map)
            blur = cv2.GaussianBlur(self.segmentation_map, (5,5), 0)
            _, thresh = cv2.threshold(blur, 253, 255, cv2.THRESH_OTSU)
            
            #eroded and dialated
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            er = cv2.erode(self.segmentation_map, kernel, iterations = 2)
            er = cv2.dilate(self.segmentation_map, kernel, iterations = 2)
#            er = cv2.morphologyEx(self.segmentation_map, cv2.MORPH_OPEN, kernel)
            
            cv2.imwrite(outdir + "/gf" + str(counter + 1) + ".png", thresh)
            cv2.imwrite(outdir + "/er" + str(counter + 1) + ".png", er)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    video_path = "data/beachVolleyball1.mov"
    test_video = "data/bgtest/car-perspective-2.avi"
    # cap = cv2.VideoCapture(video_path)

    vbg = VibeBG(req_matches = 2)
    # vbg.initialize_bg(cap.read()[1])
    # for num, bg in enumerate(vbg.samples):
    #     cv2.imshow(str(num), bg)
    # cv2.waitKey(0)

    frames = bt_util.load_frame_folder("data/deduped_beachVolleyball1")
    print("Loaded frames")
    vbg.pipeline_frames(frames)

    # vbg.pipeline(cv2.VideoCapture(test_video), outdir = "data/tests/bg1")








