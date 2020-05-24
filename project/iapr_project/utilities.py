#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper functions
"""

import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

from skimage.morphology import remove_small_objects, binary_closing
from skimage.filters import threshold_yen, unsharp_mask
from skimage.exposure import rescale_intensity


def compute_angle(a, b, c):
    """Compute angle of point b given point a and c.

    Args:
        a, b, c: Three points.
    Return:
        float: Angles of abc.
    """
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


def compute_elongation(m):
    """Compute elongation given the moments.

    Args:
        m: Moments.
    Return:
        float: elongation.
    """
    x = m['mu20'] + m['mu02']
    y = 4 * m['mu11'] ** 2 + (m['mu20'] - m['mu02']) ** 2
    return (x + y ** 0.5) / (x - y ** 0.5)


def parse_equation(equations):
    """Parse the classification results.

    Args:
        equations: List of classification results (class indices).
    Return:
        string: Math equation.
    """
    str_eq = ''
    eq_map = { 0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
               9: '+', 10: '-', 11: '*', 12: '/', 13: '=' }
    for e in equations:
        str_eq += eq_map[e]

    if len(equations) > 1:
        if equations[-1] == 13:
            ans = eval(str_eq[:-1])
            str_eq += str(ans)

    return str_eq


def put_information(image, frame_string, eq_string):
    """Put information on the frame.

    Args:
        image: The image to put information on.
        frame_string: A string of frame number.
        eq_string: A string of the current equation.
    """
    frame_text_size, _ = cv2.getTextSize(frame_string, cv2.FONT_HERSHEY_DUPLEX, 1, 1)
    eq_text_size, _ = cv2.getTextSize(eq_string, cv2.FONT_HERSHEY_DUPLEX, 1, 1)
    # draw black backgrounds
    cv2.rectangle(image, (0, 480 - frame_text_size[1] - eq_text_size[1] - 30), (frame_text_size[0], 480), (0, 0, 0), -1)
    cv2.rectangle(image, (0, 480 - eq_text_size[1] - 20), (eq_text_size[0], 480), (0, 0, 0), -1)
    # write information
    cv2.putText(image, frame_string, (0, 480 - frame_text_size[1] - 25), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, eq_string, (0, 480 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)


def imshow(image, cmap='viridis'):
    """Plot the image.

    Args:
        image: The image that will be plotted.
        cmap: Color map.
    """
    plt.figure(figsize=(9, 6))
    plt.imshow(image, cmap=cmap)
    plt.show()


def mask_image(image, lower1, upper1, lower2, upper2):
    """Mask the image given color ranges.

    Args:
        image: The image that will be masked.
        lower1: 1st Lower color range.
        upper1: 1st Upper color range.
        lower2: 2nd Lower color range.
        upper2: 2nd Upper color range.
    Return:
        numpy.ndarray: The masked image.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = mask1 + mask2
    res = cv2.bitwise_and(image, image, mask=mask)

    return res


def preprocess(image, min_size, threshold=True):
    """Preprocess the image.

    Args:
        image: The input image.
        min_size: Minimum size to be kept.
        threshold: Whether to apply thresholding.
    Return:
        numpy.ndarray: The preprocessed image.
    """
    # convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # apply thresholding
    if threshold:
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # remove small noises using morphology
    image = remove_small_objects(image.astype(bool), min_size=min_size)
    # fill holes using morphology
    image = binary_closing(image).astype(np.uint8)

    return image


def normalize(image):
    """Normalize the image to handle illumination differences.

    Args:
        image (numpy.ndarray): The input image.
    Return:
        numpy.ndarray: A normalized image in terms of intensity.
    """
    # convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # sharpen image
    image = unsharp_mask(image, radius=5, amount=4, preserve_range=True)
    # get the threshold using Yen's method
    yen_threshold = threshold_yen(image)
    # rescale the intensity with the threshold
    image = rescale_intensity(image, (0, yen_threshold), (0, 255))

    return image
