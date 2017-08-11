import os
import src.lib.tools as tl
import numpy as np
import cv2


def run_main():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tf2.mp4')
    # noinspection PyArgumentList
    cap = cv2.VideoCapture(path)

    first_roi = None
    prev_kp, prev_des = None, None

    while True:
        ret, frame = cap.read()

        # Set the ROI (Region of Interest). Actually, this is a
        # rectangle that we're tracking
        c, r, w, h = 1270, 129, 50, 80
        roi = cv2.cvtColor(frame[r:r + h, c:c + w], cv2.COLOR_RGB2GRAY)
        # roi = frame[r:r + h, c:c + w]

        if first_roi is None:
            first_roi = roi.copy()
            # Initiate STAR detector
            sift = cv2.xfeatures2d.SIFT_create(nfeatures=80)
            prev_kp, prev_des = sift.detectAndCompute(first_roi, None)

        # Initiate FAST object with default values
        fast = cv2.FastFeatureDetector_create(threshold=100)  # <--- Use it. It`s  Accurate

        # find and draw the keypoints
        fast_kp = fast.detect(roi)  # compute keypoints
        fast_featured_roi = cv2.drawKeypoints(roi, fast_kp, None, color=(255, 0, 0))

        # Initiate STAR detector
        orb = cv2.ORB_create()

        # find the keypoints with ORB
        orb_kp = orb.detect(roi, None)
        orb_featured_roi = cv2.drawKeypoints(roi, orb_kp, None, color=(0, 255, 0), flags=0)

        good = cv2.goodFeaturesToTrack(roi, 100, 0.3, 7, blockSize=7)
        corners = np.float32(good)
        good_roi = roi.copy()

        for item in corners:
            x, y = item[0]
            cv2.circle(good_roi, (x, y), 2, 255, -1)

        # Initiate STAR detector
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=80)

        # find the keypoints with ORB
        sift_kp = sift.detect(roi, None)
        sift_featured_roi = cv2.drawKeypoints(roi, sift_kp, None, color=(0, 255, 0), flags=0)

        kp, des = sift.detectAndCompute(roi, None)  # compute keypoints

        # BFMatcher
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(prev_des, des, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        # Draw first 10 matches.
        roi_matches = cv2.drawMatchesKnn(first_roi, prev_kp, roi, kp, good, None, flags=2)

        multiply_roi = cv2.addWeighted(first_roi, 0.7, roi, 0.3, 0)

        cv2.imshow('Tracking', tl.concat_hor((
                                              cv2.resize(roi, (0, 0), fx=3, fy=3),
                                              cv2.resize(multiply_roi, (0, 0), fx=3, fy=3),
                                              cv2.resize(fast_featured_roi, (0, 0), fx=3, fy=3),
                                              cv2.resize(good_roi, (0, 0), fx=3, fy=3),
                                              cv2.resize(sift_featured_roi, (0, 0), fx=3, fy=3),
                                              cv2.resize(roi_matches, (0, 0), fx=3, fy=3),
                                              )))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_main()
