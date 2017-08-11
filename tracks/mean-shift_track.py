import os
import src.lib.tools as tl
import numpy as np
import cv2


def run_main():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tf2.mp4')
    # noinspection PyArgumentList
    cap = cv2.VideoCapture(path)
    # Read the first frame of the video
    ret, frame = cap.read()
    # Set the ROI (Region of Interest). Actually, this is a
    # rectangle that we're tracking
    c, r, w, h = 1283, 149, 20, 40
    track_window = (c, r, w, h)
    # Create mask and normalized histogram
    roi = frame[r:r + h, c:c + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_roi, np.array([0., 30., 32.]), np.array([180., 255., 255.]))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        x, y, w, h = track_window
        cv2.rectangle(frame, (c, r), (c + w, r + h), 0, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        cv2.putText(frame, 'Tracked', (x - 25, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Tracking', tl.concat_hor((cv2.resize(frame, (0, 0), fx=0.5, fy=0.5), roi_hist)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_main()
