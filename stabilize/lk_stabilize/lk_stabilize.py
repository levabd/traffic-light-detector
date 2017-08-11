#!/usr/bin/env python

"""
Keys
----
ESC - exit
"""

# Python 2/3 compatibility
import os

import cv2
import numpy as np

import tools as tl

lk_params = dict(winSize=(10, 10), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=1, blockSize=7)


class App:
    def __init__(self, video_src):
        self.track_len = 50
        self.detect_interval = 10
        # noinspection PyArgumentList
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0
        self.prev_gray = None
        self.prev_transform = np.float32([[1, 0, 0], [0, 1, 0]])
        self.transform_for_saving = np.float32([[1, 0, 0], [0, 1, 0]])
        self.prev_features = None

    def run(self):
        # c, r, w, h = 1230, 100, 120, 140
        c, r, w, h = 1270, 129, 50, 80

        while True:
            ret, frame = self.cam.read()
            stab_vis = frame.copy()
            crop_frame = frame[r:r + h, c:c + w]
            crop_stab_vis = crop_frame.copy()
            bad_stab_vis = crop_frame.copy()
            frame_gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
            vis = crop_frame.copy()

            if self.prev_features is not None:
                pc, stc, errc = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, self.prev_features, None, **lk_params)
                transform = cv2.estimateRigidTransform(pc, self.prev_features, False)
                if transform is not None:
                    # I don`t want to scale and rotate image with Affine transformation
                    transform[0, 0] = 1  # scale X
                    transform[1, 1] = 1  # scale Y
                    transform[0, 1] = 0  # rotate X
                    transform[1, 0] = 0  # rotate Y

                    # Multiply previous and current transform
                    transform[0, 2] += self.prev_transform[0, 2]  # dX
                    transform[1, 2] += self.prev_transform[1, 2]  # dY

                    self.transform_for_saving = transform
                    rows, cols, cl = frame.shape
                    rows2, cols2 = frame_gray.shape
                    stab_vis = cv2.warpAffine(frame, transform, (cols, rows))
                    bad_stab_vis = cv2.warpAffine(crop_frame, transform, (cols2, rows2))
                    crop_stab_vis = stab_vis[r:r + h, c:c + w]

                p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, self.prev_features, None, **lk_params)

                for (x, y) in p1.reshape(-1, 2):
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)

                for (x, y) in self.prev_features.reshape(-1, 2):
                    cv2.circle(vis, (x, y), 2, (255, 0, 0), -1)

            if self.frame_idx % self.detect_interval == 0:
                self.prev_gray = frame_gray
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                # save previous interval transform
                self.prev_transform = self.transform_for_saving
                # Correction
                self.prev_transform[0, 2] = self.prev_transform[0, 2] if abs(self.prev_transform[0, 2]) < 10 else 0
                self.prev_transform[1, 2] = self.prev_transform[1, 2] if abs(self.prev_transform[1, 2]) < 10 else 0
                self.prev_features = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)

            self.frame_idx += 1
            cv2.imshow('lk_track', tl.concat_hor((
                                                    cv2.resize(vis, (0, 0), fx=3, fy=3),
                                                    cv2.resize(bad_stab_vis, (0, 0), fx=3, fy=3),
                                                    cv2.resize(crop_stab_vis, (0, 0), fx=3, fy=3),
                                                    cv2.resize(stab_vis, (0, 0), fx=0.5, fy=0.5)
                                                )))
            # cv2.imshow('lk_track', vis)

            ch = cv2.waitKey(1)
            if ch == 27:
                break


def main():
    # noinspection PyBroadException
    try:
        video_src = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../tf2.mp4')
    except:
        video_src = 0

    print(__doc__)
    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
