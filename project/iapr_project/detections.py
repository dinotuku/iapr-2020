#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Object detection functions
"""

import cv2
import numpy as np

from utilities import compute_angle, compute_elongation, imshow, mask_image, preprocess


def find_red_arrow(image, show=False):
    """Detect the red arrow in the image.

    Args:
        image: The input image.
        show: Whether to show the results.
    Return:
        tuple: Tip coordinates.
        tuple: Center coordinates.
    """
    image_copy = image.copy()

    masked = mask_image(image_copy,
        np.array([0, 100, 0]), np.array([20, 255, 255]),
        np.array([160, 100, 0]), np.array([180, 255, 255]))
    preprocessed = preprocess(masked, 100)

    contours, _ = cv2.findContours(preprocessed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # compute the centroid of the shapes
        M = cv2.moments(c)

        area = M['m00']
        elongation = compute_elongation(M)
        # these will not be the arrow (too small or too big)
        if area < 1000 or area > 10000 or elongation > 100: continue

        cX = int(M['m10'] / area)
        cY = int(M['m01'] / area)
        center = (cX, cY)

        # Not sure do we need this
        # if abs(M['mu20'] - M['mu02']) > 420000: continue

        # find the corners of the arrow
        points = cv2.approxPolyDP(c, 4.7, True).squeeze(1)

        tip_idx = 0
        cand_tips = []
        angles = []

        # find tip candidates
        for i in range(len(points)):
            # get the current point and the surrounding points
            x = points[i - 1] if i != 0 else points[-1]
            y = points[i]
            z = points[i + 1] if i != len(points) - 1 else points[0]
            # get the lengths between the current point and the surrounding points
            l1 = np.linalg.norm(np.array(x) - np.array(y))
            l2 = np.linalg.norm(np.array(y) - np.array(z))

            ang = compute_angle(x, y, z)
            angles.append(ang)
            # save candidates
            if abs(ang - 100) < 15 and (l1 + l2 > 30):
                cand_tips.append(len(angles) - 1)

        # choose the correct tip
        for i in cand_tips:
            pang = angles[i - 1] if i != 0 else angles[-1]
            nang = angles[i + 1] if i != len(angles) - 1 else angles[0]
            if pang + nang < 300 and pang + nang > 200:
                tip_idx = i

        # visualize the result on the image
        cv2.drawContours(image_copy, [c], 0, (214, 39, 40), 2)
        cv2.circle(image_copy, tuple(center), 5, (0, 255, 0), -1)
        cv2.circle(image_copy, tuple(points[tip_idx]), 5, (0, 0, 255), -1)

        break

    if show:
        imshow(image_copy)

    return points[tip_idx], center


def find_math_elements(image, arrow_c, bound=20, show=False):
    """Detect math elements in the image.

    Args:
        image: The input image.
        arrow: Center of the arrow.
        bound: Bounding box size.
        show: Whether to show the results.
    Return:
        list: Images of all math elements.
        list: Center coordinates of math elements.
    """
    image_original = image.copy()
    image_copy = image.copy()

    # cover red arrow with white rectangle
    cv2.rectangle(image_copy, (arrow_c[0] - 60, arrow_c[1] - 60), (arrow_c[0] + 60, arrow_c[1] + 60), (255, 255, 255), -1)

    value_threshold = int(cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 2].mean() * 0.9)
    masked = mask_image(image_copy,
        np.array([0, 0, 0]), np.array([180, 255, value_threshold]),
        np.array([100, 100, 0]), np.array([140, 255, 255]))
    preprocessed = preprocess(masked, 10, False)

    # get the contours of all shapes
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    elements = []

    for i, c in enumerate(contours):
        # compute the centroid of the shapes
        M = cv2.moments(c)

        area = M['m00']
        elongation = compute_elongation(M)
        # these are either too small or too big or too elongated
        if area < 40 or area > 400 or elongation > 3000: continue

        cY = int(M['m01'] / M['m00'])
        cX = int(M['m10'] / M['m00'])
        center = (cX, cY)

        # if it is too close to a known element, it is not a valid element
        too_close = False
        for center_ in centers:
            d = (center_[0] - center[0]) ** 2 + (center_[1] - center[1]) ** 2
            if d < 4000:
                too_close = True
                break
        if too_close: continue

        # save element and center
        element = image[cY - bound: cY + bound, cX - bound:cX + bound]
        element = cv2.resize(element, (28, 28))
        elements.append(element)
        centers.append(center)

        # visualize the result on the image
        label_color = (214, 39, 40)
        cv2.rectangle(image_original, (cX - bound, cY - bound), (cX + bound, cY + bound), label_color, 2)
        cv2.putText(image_original, f'{len(elements) - 1}', (cX, cY + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

    if show:
        imshow(image_original)

    return elements, centers
