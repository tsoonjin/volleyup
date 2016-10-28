#!/usr/bin/env python
import sys
sys.path.insert(0, '../')

from volleyup.utils.Video import Video
from volleyup.utils.utils import get_basename

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        print("Usage: python extract_frames.py VIDEO DIR_PATH\n \
              VIDEO: \t\tpath of video that wished to be extracted\n \
              DIR_PATH: \tstores extracted frames from video. Current directory used by default.")
        exit()
    else:
        vidpath = args[0]
        vidname = get_basename(vidpath)
        dirpath = args[1] if len(args) > 1 else vidname
        skip = int(args[2]) if len(args) > 2 else 0
        video = Video(vidpath)
        print("Processing video ...")
        video.write_frames(dirpath, skip=skip)
