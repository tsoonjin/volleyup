import numpy as np
import cv2
import os

import bt_util

from flow import LKTracker

class FlowBg:
	def __init__(self):
		self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
		self.st_params = dict( maxCorners = 100,
		                        qualityLevel = 0.3,
		                        minDistance = 7,
		                        blockSize = 7 )
	def get_flow(self, image1, image2):
		pass

	def pipeline_frames(self, frames):
		width, height, *_ = frames[0].shape
		frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
		f0 = frames[0]
		p0 = cv2.goodFeaturesToTrack(f0, mask = None, **self.st_params)
		for frame in frames[1:]:
			f1 = frame
			p1, st, err = cv2.calcOpticalFlowPyrLK(f0, f1, p0, None, **self.lk_params)
			good_new = p1[st == 1]
			good_old = p0[st == 1]
			change = [new - old for new, old in zip(good_new, good_old)]
			dx = int(np.round(np.median([i[0] for i in change])))
			dy = int(np.round(np.median([i[1] for i in change])))
			bg = np.zeros_like(frame)
			for x in range(width):
				for y in range(height):
					nx, ny = x + dx, y + dy
					print (nx, width, ny, height )
					if nx >= width or nx < 0:
						continue
					if ny >= height or ny < 0:
						continue
					bg[nx][ny] = f0[x][y]
			res = f1 - bg
			cv2.imshow("subtracted", res)
			cv2.waitKey(50)
			f0 = f1
			p0 = p1



print(os.getcwd())
video_folder = "data"
video_paths = ["/".join([video_folder, video_path]) for video_path in os.listdir(video_folder) if video_path.endswith(".mov")]
lkt = LKTracker(cv2.VideoCapture(video_paths[0]))
lkt.run()

# frames = bt_util.load_frame_folder("data/deduped_beachVolleyball1")
# fbg = FlowBg()
# fbg.pipeline_frames(frames)
