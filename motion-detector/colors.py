import numpy as np
import cv2


lower_u_red = np.array([150, 0, 0])
greater_u_red = np.array([180, 255, 255])

# lower_red = np.array([15, 0, 0])
lower_red = np.array([0, 0, 0])
greater_red = np.array([37, 255, 255])

lower_green = np.array([20, 0, 0])
greater_green = np.array([150, 255, 255])

red = np.array([0, 100, 100])
green = np.array([120, 100, 100])


def red_or_green(image, c, candidate):
    # construct a mask for the contour, then compute the
    # average value for the masked region
    mask_pixels = []
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    pts = np.where(mask == 255)
    mask_pixels.append(image[pts])
    mean = cv2.mean(image, mask=mask)[:3]
    # median = np.uint8([[[np.median(mask_pixels, 0), np.median(mask_pixels, 1), np.median(mask_pixels, 2)]]])
    # print median
    h_mean = cv2.cvtColor(np.uint8([[[mean[0], mean[1], mean[2]]]]), cv2.COLOR_BGR2HSV)
    # hsv_median = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
    print h_mean
    # print hsv_median

    if candidate == "green":
        if np.count_nonzero(h_mean > lower_green) == 3 and np.count_nonzero(greater_green > h_mean) == 3:
            return True
        else:
            return False

    if candidate == "red":
        if (np.count_nonzero(h_mean > lower_red) == 3 and np.count_nonzero(greater_red > h_mean) == 3) \
                or (np.count_nonzero(h_mean > lower_u_red) == 3 and np.count_nonzero(greater_u_red > h_mean) == 3):
            return True
        else:
            return False
