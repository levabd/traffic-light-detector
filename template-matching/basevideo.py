"""
Keys
----
ESC - exit
"""
import cv2


class BaseVideo:

    def __init__(self, video_src, roi=None, first_frame=0):
        # noinspection PyArgumentList
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0
        self.first_frame = first_frame
        self.frame_width = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_height = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if roi is None:
            self.c, self.r, self.w, self.h = 0, 0, int(self.frame_width), int(self.frame_height)
        else:
            self.c, self.r, self.w, self.h = roi[2], roi[0], roi[3] - roi[2], roi[1] - roi[0]

    def prepare(self):
        """
        Called one time in run function before video reading started
        """
        pass

    def proceed_frame(self, frame):
        """
        Called every frame. Video processing
        :param frame: current frame
        """
        pass

    # noinspection PyMethodMayBeStatic
    def show_debug_images(self, frame):
        """
        :param frame: current frame
        :return: images on the left side of debugging
        """
        return frame

    def run(self):
        self.cam.set(1, self.first_frame)

        self.prepare()

        while True:
            ret, frame = self.cam.read()

            if frame is None:  # Repeat
                self.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            self.proceed_frame(frame)

            self.frame_idx += 1

            cv2.imshow('Result', self.show_debug_images(frame))

            ch = cv2.waitKey(1)
            if ch == 27:
                break
