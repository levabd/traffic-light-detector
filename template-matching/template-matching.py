#!/usr/bin/env python

# Python 2/3 compatibility
import os

import cv2
import numpy as np

import tools as tl
from basevideo import BaseVideo


class App(BaseVideo):

    def __init__(self, video_src, roi):
        BaseVideo.__init__(self, video_src, roi=roi)
        self.template = None
        self.frame_gray = None
        self.out_frame = None
        self.threshold = 0.9

    def prepare(self):
        ret, frame = self.cam.read()
        self.template = frame[self.r:self.r + self.h, self.c:self.c + self.w]
        self.template = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        pass

    def proceed_frame(self, frame):
        self.out_frame = frame.copy()
        self.frame_gray = cv2.cvtColor(self.out_frame, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(self.frame_gray, self.template, cv2.TM_CCOEFF_NORMED)
        location = np.where(result >= self.threshold)

        for point in zip(*location[::-1]):
            cv2.rectangle(self.out_frame, point, (point[0] + self.w, point[1] + self.h), (0, 255, 255), 2)

    # noinspection PyMethodMayBeStatic
    def show_debug_images(self, frame):
        return tl.concat_hor((
            self.template,
            cv2.resize(self.out_frame, (0, 0), fx=0.5, fy=0.5),
            cv2.resize(self.frame_gray, (0, 0), fx=0.5, fy=0.5)))


def main():
    # noinspection PyBroadException
    try:
        video_src = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tf2.mp4')
    except:
        video_src = 0

    print(__doc__)
    App(video_src=video_src, roi=[184, 198, 1280, 1296]).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()