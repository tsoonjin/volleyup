import cv2
import os
import numpy as np
from collections import deque

from Video import Video
from tracker import median_bg_sub
from collections import Counter

print("cwd: ", os.getcwd())
video_folder = "../../beachVolleyball"
video_paths = [os.path.join(video_folder, video_path) for video_path in os.listdir(video_folder)]

class VTester(Video):
    def __init__(self, name, bg_hist = 20, bg_thresh = 30):
        super().__init__(name, bg_hist, bg_thresh)

    def load_frames(self, funct = lambda x: x, dedupe = 0):
        """
        loads frames of video into memory for downstream processing
        """
        self.reset_video()
        self.frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Breaking")
                break
            if dedupe:
                if self.frames:
                    sub = abs(self.frames[-1] - funct(frame))
                    sub = sub < 3
                    identity = Counter(sub.ravel())
                    # print(identity)
                    if identity[False] < identity[True]:
                        continue
            #         cv2.imshow("testdupe", frame)
            # if cv2.waitKey(500) & 0xFF == ord('q'):
            #     break
            self.frames.append(funct(frame))
        return self.frames
    def dump_frames(self, frames, folder_path):
        os.makedirs(folder_path)
        for num, frame in enumerate(frames):
            cv2.imwrite(os.path.join(folder_path, str(num)) + ".png", frame)

    def pipeline(self):
        pass

def median_fg_mask(frames, bg_hist=3, thresh_limit=60):
    """ Performs background subtraction via frame differencing with median of previous n frames
    Parameters
    ----------
    bg_hist      : number of frames before current frame considerd as background
    thresh_limit : threshold to differentiate foreground and background

    """
    bg_frames = deque([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames[0:bg_hist]],
                      maxlen=bg_hist)
    res = []
    for f in frames[bg_hist:]:
        curr = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        bg = np.median(np.array(list(bg_frames)), axis=0)
        diff = np.fabs(curr - bg)
        mask = cv2.threshold(np.uint8(diff), thresh_limit, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
        bg_frames.append(curr)
        res.append(mask)
    return res

def load_frame_folder(folder_path):
    frames = []
    for file_path in os.listdir(folder_path):
        full = os.path.join(folder_path, file_path)
        frames.append((cv2.imread(full), int(file_path.split(".")[0])))
    return [i[0] for i in sorted(frames, key = lambda x: x[1])]

def ballfinder(frames, bg_hist =3, thresh_limit = 60):
    bg_frames = deque([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames[0:bg_hist]],
                      maxlen=bg_hist)
    for f in frames[bg_hist:]:
        
        curr = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        bg = np.median(np.array(list(bg_frames)), axis=0)
        diff = np.fabs(curr - bg)
        mask = cv2.threshold(np.uint8(diff), thresh_limit, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        canvas = np.zeros_like(f)
        e_canvas = f.copy()
        combined = f.copy()
        if len(cnts) >= 1:
            cnts.sort(key=cv2.contourArea, reverse=True)
            for cnt in cnts:
                cnt_area =  cv2.contourArea(cnt)
                if cnt_area > 80:
                    continue
                cv2.drawContours(canvas, [cnt], -1, (255, 0, 0), 1)

                cv2.drawContours(combined, [cnt], -1, (255, 0, 0), 1)
                try:
                    ellipse = cv2.fitEllipse (cnt)
                    print(ellipse)
                    cv2.ellipse(e_canvas, ellipse, (255, 0, 0), 1)
                except Exception as err:
                    print(err)
        bg_frames.append(curr)
        cv2.imshow("original", f)
        cv2.imshow("mask", mask)
        cv2.imshow("contorus", canvas)
        cv2.imshow("combined", combined)
        cv2.imshow("e_canvas", e_canvas )
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break



def bbg_subtractor(frames, thresh = 0.5):
    """
    Performs bg_subtraction on a grayscale image
    """
    for i in range(1, len(frames-1)):
        a, b, c = frames[i-1], frames[i], frames[i+1]
        Dba, Dbc = b-a, b-c
        ret, Dba = cv2.threshold(Dba, int(thresh * 255), 255, cv2.THRESH_BINARY)
        ret, Dbc = cv2.threshold(Dba, int(thresh * 255), 255, cv2.THRESH_BINARY)




# video_paths = video_paths[0]
# output_folder = "data/deduped_"+os.path.splitext(os.path.split(video_paths)[1])[0]
# v = VTester(video_paths)

# frames = v.load_frames(dedupe = 1)
# v.dump_frames(frames, output_folder)
# frames = load_frame_folder(output_folder)
# ballfinder(frames)

# frames = median_fg_mask(frames)
# print(len(frames))
# print(frames[0])
# for frame in frames:
#     cv2.imshow("frame", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

target = "data/bgtest/car-perspective-2.avi"
assert(os.path.exists(target))
a = cv2.VideoCapture(target)
while True:
    ret, frame = a.read()
    if not ret:
        break
    cv2.imshow("test", frame)
    cv2.waitKey(50)









