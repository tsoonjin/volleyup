import os
import cv2

def load_frame_folder(folder_path):
    frames = []
    for file_path in os.listdir(folder_path):
        full = os.path.join(folder_path, file_path)
        frames.append((cv2.imread(full), int(file_path.split(".")[0])))
    return [i[0] for i in sorted(frames, key = lambda x: x[1])]