#!/usr/bin/env python
import os
import cv2
import numpy as np

import config

from utils import get_video_source
from flow import LKTracker, FarnebackTracker


class Video():
    """ Represents raw video for further analysis
    Parameters
    ----------
    bg_hist     : number of frames accumulated as background
    bg_thresh   : minimum difference between current frame and background
    """
    def __init__(self, name, bg_hist=20, bg_thresh=30):
        self.name = os.path.splitext(name.rsplit('/', 1)[-1])[0]
        self.cap = get_video_source(name)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # Video intrinsic properties
        self.shape = (self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),       # (x, y)
                      self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize various components of video pipeline
        self.lk_tracker = LKTracker(self.cap)
        self.farneback_tracker = FarnebackTracker(self.cap)

    def play(self):
        self.reset_video()  # Ensures always playing video from first frame
        print("Playing {}:\t Size: {}\t Num of Frames: {}\t FPS:{}".format(
            self.name, self.shape, len(self.frames), self.fps))
        while True:
            ret, frame = self.cap.read()
            cv2.imshow(self.name, frame)
            k = cv2.waitKey(1)
            if self.is_eov():
                break
            if k == 27:
                print("Terminated by user")
                exit()
        self.cap.release()
        cv2.destroyAllWindows()

    def reset_video(self):
        """ Reset video to first frame """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def write_frames(self, dirpath=None, extension='jpg'):
        """ Write raw frames to directory given
        Parameters
        ----------
        dirpaths : directory path to save extracted images
        type     : image extension
        """
        dirpath = dirpath if dirpath else '{}{}'.format(config.DATA_DIR, self.name)
        # Create directory if does not exists
        if not os.path.exists(os.path.abspath(dirpath)):
            print("Creating directory: {}".format(os.path.abspath(dirpath)))
            os.makedirs(dirpath)
            self.reset_video()
            while True:
                frame_id = self.cap.get(cv2.CAP_PROP_POS_FRAMES) + 1
                ret, frame = self.cap.read()
                cv2.imwrite("{}/{}.{}".format(dirpath, frame_id, extension), frame)
                if self.is_eov():
                    break
            print('Successfully written frames')
        else:
            print('Directory is not empty')

    def is_eov(self):
        """ Check for end of frame in a video """
        return self.cap.get(cv2.CAP_PROP_POS_FRAMES) == self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

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
