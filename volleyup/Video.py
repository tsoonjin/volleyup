#!/usr/bin/env python
import os
import cv2
import glob
import numpy as np
from collections import deque
from utils import get_video_source
from imgproc import canny_edge
from flow import LKTracker, FarnebackTracker


class Video():
    """ Represents raw video for further analysis
    Parameters
    ----------
    bg_hist     : number of frames accumulated as background
    bg_thresh   : minimum difference between current frame and background
    desired_fps : rate at which video wished to be extracted
    """
    def __init__(self, name, bg_hist=20, bg_thresh=30, desired_fps=40):
        self.name = os.path.splitext(name.rsplit('/', 1)[-1])[0]
        self.__cap = get_video_source(name)
        self.fps = self.__cap.get(cv2.CAP_PROP_FPS)
        self.bg_hist = bg_hist
        self.bg_thresh = bg_thresh
        self.desired_fps = desired_fps
        self.frames = self.get_frames(desired_fps)
        # Video intrinsic properties
        self.shape = (self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH),       # (x, y)
                      self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize various components of video pipeline
        self.lk_tracker = LKTracker(self.__cap)
        self.farneback_tracker = FarnebackTracker(self.__cap)

    def play(self):
        self.reset_video()  # Ensures always playing video from first frame
        bg_frames = deque(maxlen=self.bg_hist)
        print("Playing {}:\t Size: {}\t Num of Frames: {}\t FPS:{}".format(
            self.name, self.shape, len(self.frames), self.fps))
        while True:
            ret, frame = self.__cap.read()
            bg_frames.append(frame)
            cv2.imshow(self.name, frame)
            k = cv2.waitKey(1)
            if self.is_eov():
                break
            if k == 27:
                print("Terminated by user")
                exit()
        self.__cap.release()
        cv2.destroyAllWindows()

    def get_frames(self, desired_fps):
        """ Extracts all frames in video given fps
        Parameters
        ----------
        desired_fps : int
            number of frames extracted per second
        """
        frames = []
        # Read straight from folder instead of iterating through video if video has been processed
        if os.path.exists(os.path.abspath("data/{}".format(self.name))):
            print("Frames already existed for {}".format(self.name))
            frames = [cv2.imread(filename)
                      for filename in glob.iglob("data/{}/*".format(self.name))]
        else:
            while True:
                ret, frame = self.__cap.read()
                if (self.__cap.get(cv2.CAP_PROP_POS_FRAMES) % self.fps) < desired_fps:
                    frames.append(frame)
                if self.is_eov():
                    break
        return frames

    def reset_video(self):
        """ Reset video to first frame """
        self.__cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def write_frames(self, dirpath='data/{}', extension='jpg'):
        """ Write raw frames to directory given
        Parameters
        ----------
        dirpaths : directory path to save extracted images
        type     : image extension
        """
        dirpath = dirpath.format(self.name)
        # Create directory if does not exists
        if not os.path.exists(os.path.abspath(dirpath)):
            print("Creating directory: {}".format(os.path.abspath(dirpath)))
            os.makedirs(dirpath)
        for i, frame in enumerate(self.frames):
            cv2.imwrite("{}/{}.{}".format(dirpath, i, extension), frame)

    def is_eov(self):
        """ Check for end of frame in a video """
        return self.__cap.get(cv2.CAP_PROP_POS_FRAMES) == self.__cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def median_frame_diff(self, curr_frame, bg_frames, thresh):
        """ Returns foreground mask via frame differencing
        Parameters
        ----------
        bg_frames : list of images that forms the background model
        thresh    : threshold limit used for separating foreground from background
        """
        if len(bg_frames) < (self.bg_hist - 1):
            return curr_frame
        median = np.median(np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in bg_frames]), axis=0)
        diff = np.fabs(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) - median)
        fg_mask = np.where(diff > thresh, np.ones(curr_frame.shape[:2]) * 255, 0)
        return cv2.cvtColor(np.uint8(fg_mask), cv2.COLOR_GRAY2BGR)

    @staticmethod
    def process_video(frames, func, wait=5):
        """ Apply function to each frame and display the image. Note that output should be BGR """
        for frame in frames:
            cv2.imshow(func.__name__, func(frame))
            k = cv2.waitKey(wait)
            if k == 27:
                exit()
