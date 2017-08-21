#!/usr/bin/env python

# Python 2/3 compatibility
import os

import cv2
import numpy as np

import tools as tl
from basevideo import BaseVideo


class App(BaseVideo):

    def __init__(self, video_src, roi):
        self.bgdModel = np.zeros((1, 65), np.float64)
        self.fgdModel = np.zeros((1, 65), np.float64)
        BaseVideo.__init__(self, video_src, roi=roi)
        self.mask = np.zeros((self.h, self.w), np.uint8)
        self.out_frame = None

    def proceed_frame(self, frame):
        frame = frame[self.r:self.r + self.h, self.c:self.c + self.w]
        cv2.grabCut(frame, self.mask, (100, 100, self.w-200, self.h-200), self.bgdModel, self.fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype('uint8')
        self.out_frame = frame * mask2[:, :, np.newaxis]

    # noinspection PyMethodMayBeStatic
    def show_debug_images(self, frame):
        frame = frame[self.r:self.r + self.h, self.c:self.c + self.w]
        return tl.concat_hor((self.out_frame, frame))


def main():
    # noinspection PyBroadException
    try:
        video_src = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tf2.mp4')
    except:
        video_src = 0

    print(__doc__)
    App(video_src=video_src, roi=[129-100, 210+100, 1270-100, 1320+100]).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()