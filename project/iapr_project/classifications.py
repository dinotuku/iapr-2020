#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classification functions
"""

import numpy as np

from utilities import normalize


def classify(image, model):
    """Classify the math digits or operators.

    Args:
        image: The input image.
        model: The trained model.
    Return:
        numpy.ndarray: The predicted probability.
        int: The predicted class.
    """
    image = 255. - normalize(image)
    image = image[np.newaxis, :, :, np.newaxis] / 255.
    prob = model.predict(image)
    pred = np.argmax(prob)

    return prob, pred
