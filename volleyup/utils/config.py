#!/usr/bin/env python

BG_DIR = 'data/bg/'
TEMPLATE_DIR = 'data/templates/'
STUB_DIR = 'data/stub/'
PANORAMA_DIR = 'data/panorama/'
DATA_DIR = 'data/'
INPUT_DATA = 'data/beachVolleyballFull.mov'
INPUT_IMGS = 'data/beachVolleyballFull'
INDVIDUAL_VIDEOS = {'{}'.format(i): 'data/beachVolleyball{}'.format(i) for i in range(1, 8)}

'''BGR COLOR CODES'''
RED = [0, 0, 255]
BLUE = [255, 128, 0]
YELLOW = [0, 255, 255]
GREEN = [0, 255, 0]
ORANGE = [0, 69, 255]
PURPLE = [204, 0, 204]
WHITE = [255, 255, 255]
BRAZIL = {'color': BLUE}
USA = {'color': RED}
LATVIA = {'color': GREEN}
ESPANYOL = {'color': ORANGE}

''' OBJECT HSV COLOR RANGES '''
OBJECT_COLOR = {'court': [(20, 0, 100), (50, 100, 255)]}

# Volleyball Court setting
FULL_COURT = (800, 400)
RESIZE_FACTOR = 0.5

# Color ranges
COLOR_RANGE = {'red': [(160, 0, 0), (180, 255, 255)], 'green': [(40, 0, 0), (80, 255, 255)],
               'white': [(0, 0, 200), (180, 30, 255)], 'sand': [(30, 0, 100), (50, 255, 255)]}

''' Video specific configurations '''
