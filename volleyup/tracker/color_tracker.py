#!/usr/bin/env python
import cv2
import numpy as np


def color_tracking(f, color, num_players=2):
    """ Performs HSV color detection with constraint on 2 players """
    mask = cv2.inRange(cv2.cvtColor(f, cv2.COLOR_BGR2HSV), color[0], color[1])
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    canvas = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    players = []
    if len(cnts) >= 1:
        cnts.sort(key=cv2.contourArea, reverse=True)
        for cnt in cnts:
            if len(players) < num_players:
                rect = cv2.minAreaRect(cnt)
                cv2.drawContours(canvas, [np.int0(cv2.boxPoints(rect))], -1, (255, 0, 0), 2)
                players.append(cnt)
            else:
                break
    return canvas
