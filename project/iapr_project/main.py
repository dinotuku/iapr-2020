#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image Analysis and Pattern Recognition - Special Project
Find and solve the math problem in a video.
"""

import argparse
import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tqdm.auto import tqdm

from classifications import classify
from detections import find_red_arrow, find_math_elements
from utilities import parse_equation, put_information

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def process_frames(frames):
    """Find the red arrow and math elements in all frames.

    Args:
        frames: Frames of the input video.
    Return:
        list: Frames of the output video.
    """
    output_frames = []
    arrow_centers = []
    preds = []
    equations = []

    # process the first frame
    frame_0 = frames[0].copy()

    # find the arrow and math elements
    last_tip, last_center = find_red_arrow(frame_0)
    elements, centers = find_math_elements(frame_0, last_center, 18)
    arrow_centers.append(last_center)
    # draw the arrow center
    cv2.circle(frame_0, tuple(last_center), 5, (255, 0, 0), -1)

    # classify math elements and save predictions
    model = load_model('models/model.h5')
    for el in elements:
        prob, pred = classify(el, model)
        preds.append(pred)

    # put text on the first frame (the robot will not be on any element in the first frame)
    put_information(frame_0, 'Frame 1', 'Equation: ')

    output_frames.append(frame_0)

    tqdm_handle = tqdm(range(len(frames) - 1), total=len(frames), initial=1)
    for i in tqdm_handle:
        # since the first frame was processed
        i += 1
        tqdm_handle.set_description(f'Frame {i + 1}')

        frame_c = frames[i].copy()

        tip, center = find_red_arrow(frame_c)
        arrow_centers.append(center)
        # plot arrow center and connect consecutive centers
        for c_idx, c in enumerate(arrow_centers):
            cv2.circle(frame_c, tuple(c), 5, (255, 0, 0), -1)
            if c_idx > 0:
                cv2.line(frame_c, tuple(arrow_centers[c_idx - 1]), tuple(c), (255, 0, 0), 1)

        # check whether the robot passes above an element
        intersect = None
        for c_i, c in enumerate(centers):
            if intersect != None: break
            for x in np.arange(0, center[0] - last_center[0], 1 if center[0] - last_center[0] > 0 else -1):
                if intersect != None: break
                for y in np.arange(0, center[1] - last_center[1], 1 if center[1] - last_center[1] > 0 else -1):
                    p = [center[0] + x, center[1] + y]
                    bound = 20
                    if ((p[0] > (c[0] - bound)) and (p[0] < (c[0] + bound))) and ((p[1] > (c[1] - bound)) and (p[1] < (c[1] + bound))):
                        intersect = c_i
                        break

        last_tip = tip
        last_center = center

        # if the robot moves too slow, don't save the same element twice
        if intersect != None:
            if len(equations) == 0:
                equations.append(preds[intersect])
            elif equations[-1] != preds[intersect]:
                equations.append(preds[intersect])

        # put information on the frame
        str_eq = parse_equation(equations)
        put_information(frame_c, f'Frame {i + 1}', f'Equation: {str_eq}')

        output_frames.append(frame_c)

    return output_frames


def main():
    """ Main function """
    parser = argparse.ArgumentParser(description='Find and solve the math problem in the video.')
    parser.add_argument('--input', type=str, help='Path to the input video.')
    parser.add_argument('--output', type=str, help='Path to save the output video.')

    args = parser.parse_args()

    # load input video frames
    print('Load input video')
    cap = cv2.VideoCapture(args.input)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    # find and solve the math equation in frames
    print('Process frames')
    output_frames = process_frames(frames)

    # write video frames
    print('Write output video')
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc('H','F','Y','U'), 2, (720, 480))
    for f in output_frames:
        f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        out.write(f)

    out.release()


if __name__ == '__main__':

    main()
