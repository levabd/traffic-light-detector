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
import colors

lk_params = dict(winSize=(10, 10), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=1, blockSize=7)

dilate_kernel = np.array([[0, 0, 1, 0, 0],
                          [0, 1, 1, 1, 0],
                          [1, 1, 1, 1, 1],
                          [0, 1, 1, 1, 0],
                          [0, 0, 1, 0, 0]], np.uint8)


class App:

    statuses = {
        0: 'Red',
        1: 'Yellow',
        2: 'Green'
    }

    candidates = {
        0: 'red',
        1: 'red',
        2: 'green'
    }

    statuses_debug_points = [
        np.array((17, 32)),
        np.array((17, 47)),
        np.array((17, 63))
    ]

    statuses_colors = {
        0: (0, 0, 255),
        1: (0, 255, 255),
        2: (0, 255, 0),
    }

    @staticmethod
    def closest_node(node, nodes):
        nodes = np.asarray(nodes)
        deltas = nodes - node
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        return np.argmin(dist_2)

    def __init__(self, video_src, frame):
        self.track_len = 50
        self.current_status = None
        self.detect_interval = 10
        # noinspection PyArgumentList
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0
        self.prev_gray = None
        self.prev_transform = np.float32([[1, 0, 0], [0, 1, 0]])
        self.transform_for_saving = np.float32([[1, 0, 0], [0, 1, 0]])
        self.prev_features = None
        self.c, self.r, self.w, self.h = frame[2], frame[0], frame[3] - frame[2], frame[1] - frame[0]
        self.contour_candidates = np.zeros((self.h, self.w, 3), np.uint8)
        self.contour_candidates[:] = (255, 255, 255)

    def run(self):
        c, r, w, h = self.c, self.r, self.w, self.h

        bg = cv2.createBackgroundSubtractorMOG2(30, 150, True)

        self.cam.set(1, 4000)

        while True:
            ret, frame = self.cam.read()

            stab_vis = frame[r - h:r + 2 * h, c - w:c + 2 * w]
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
                    rows, cols, cl = stab_vis.shape
                    rows2, cols2 = frame_gray.shape
                    stab_vis = cv2.warpAffine(stab_vis, transform, (cols, rows))
                    bad_stab_vis = cv2.warpAffine(crop_frame, transform, (cols2, rows2))
                    crop_stab_vis = stab_vis[h:h + h, w:w + w]

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
            blurred_stab_vis = cv2.GaussianBlur(crop_stab_vis, (5, 5), 10)
            stab_gray = cv2.cvtColor(blurred_stab_vis, cv2.COLOR_BGR2GRAY)
            bg_mask = bg.apply(stab_gray)

            # Morph open the thresholded image to fill in holes, then find contours on thresholded image
            kernel = np.ones((5, 5), np.uint8)
            bg_mask_thresh = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel)
            _, cnts, _ = cv2.findContours(bg_mask_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            crop_stab_contour = blurred_stab_vis.copy()

            contour_candidates2 = np.zeros((self.h, self.w, 3), np.uint8)

            # loop over the contours
            contour_area = 0

            for contour in cnts:
                (x, y, cw, ch) = cv2.boundingRect(contour)
                ar = cw / float(ch)

                tmp_contour_area = cv2.contourArea(contour)

                # if the contour is too rectangle (not square), ignore it
                if ar < 0.85 or ar > 1.45 or tmp_contour_area / cv2.arcLength(contour, True) < 1.5:
                    continue

                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                center = (int(cx), int(cy))
                radius = int(radius)

                # if the contour is too small, ignore it
                # if radius > 15 or radius < 7:
                #     continue

                # compute the bounding circle for the contour, draw it on the frame,
                cv2.circle(crop_stab_contour, center, radius, (0, 255, 0), 2)
                cv2.rectangle(crop_stab_contour, (x, y), (x + cw, y + ch), (0, 0, 255), 2)

                # "Smart" Yellow light recogniser
                # green_distance = np.linalg.norm(np.array((int(cx), int(cy)))-self.statuses_debug_points[1])
                # red_distance = np.linalg.norm(np.array((int(cx), int(cy)))-self.statuses_debug_points[0])
                # s_c = 1 if green_distance < red_distance else 0

                s_c = self.closest_node(np.array((int(cx), int(cy))), self.statuses_debug_points)

                if tmp_contour_area > contour_area:
                    # Greater contour in this frame
                    contour_area = tmp_contour_area

                    # Double check
                    if colors.red_or_green(blurred_stab_vis, contour, self.candidates[s_c]):
                        self.current_status = s_c
                        # reinit Background Subtractor
                        bg = cv2.createBackgroundSubtractorMOG2(30, 150, True)
                        # full_distance = np.linalg.norm(self.statuses_debug_points[0] - self.statuses_debug_points[1])
                        # if self.current_status == 0 and red_distance > full_distance / 3:
                        #     status_candidate = 2
                        cv2.circle(self.contour_candidates, center, 1, self.statuses_colors[self.current_status], 1)
                        cv2.circle(self.contour_candidates, center, radius, self.statuses_colors[self.current_status], 1)

                print "Status {} on frame {}".format(self.statuses[s_c], self.frame_idx)
                print center
                print tmp_contour_area
                print tmp_contour_area / cv2.arcLength(contour, True)
                print colors.red_or_green(blurred_stab_vis, contour, self.candidates[s_c])

                contour_candidates2 = cv2.bitwise_and(blurred_stab_vis, blurred_stab_vis, mask=bg_mask_thresh)

            if self.current_status is not None:
                cv2.putText(frame, self.statuses[self.current_status], (450, 500), cv2.FONT_HERSHEY_SIMPLEX, 10,
                            self.statuses_colors[self.current_status], 20)

            cv2.imshow('lk_track',
                       tl.concat_hor((
                           tl.concat_ver((
                               tl.concat_hor((
                                   cv2.resize(vis, (0, 0), fx=3, fy=3),
                                   cv2.resize(bad_stab_vis, (0, 0), fx=3, fy=3),
                                   stab_vis,
                                   cv2.resize(crop_stab_vis, (0, 0), fx=3, fy=3)
                               )),
                               tl.concat_hor((
                                   cv2.resize(bg_mask, (0, 0), fx=3, fy=3),
                                   cv2.resize(bg_mask_thresh, (0, 0), fx=3, fy=3),
                                   cv2.resize(crop_stab_contour, (0, 0), fx=3, fy=3),
                                   cv2.resize(contour_candidates2, (0, 0), fx=3, fy=3)
                                   # cv2.resize(self.contour_candidates, (0, 0), fx=3, fy=3)
                               ))
                           )),
                           cv2.resize(frame, (0, 0), fx=0.5, fy=0.5),
                       ))
                       )

            ch = cv2.waitKey(1)
            if ch == 27:
                break


def main():
    # noinspection PyBroadException
    try:
        video_src = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tf2.mp4')
    except:
        video_src = 0

    print(__doc__)
    App(video_src=video_src, frame=[129, 210, 1270, 1320]).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
