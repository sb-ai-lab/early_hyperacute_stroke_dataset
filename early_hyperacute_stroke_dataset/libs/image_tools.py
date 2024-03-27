import cv2
import numpy as np


def float_image_to_grayscale(image, as_bgr=False):
    min_val = image.min()
    max_val = image.max()

    k = 255.0 / (max_val - min_val)
    b = -k * min_val

    image = (k * image + b).astype(np.uint8)

    if as_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def apply_windowing_by_setup(image, center, width):
    if center is not None and width is not None:
        image_min = center - width / 2
        image_max = center + width / 2

        image[image < image_min] = image_min
        image[image > image_max] = image_max

    return image
