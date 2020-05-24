#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training the digits/operators classifier
"""

import argparse
import datetime
import os
import pathlib

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import backend as K

from detections import find_red_arrow, find_math_elements
from utilities import normalize


def load_data(op_path, vid_path):
    """Load training data from MNIST, the given operators image and the digits/operators in the video.

    Args:
        op_path: Path to the operator image.
        vid_path: Path to the video.

    Return:
        tuple: training data.
        tuple: testing data.
    """
    # load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # load the given operators image, crop it for each operator and generate more data
    operators = cv2.imread(op_path)
    operators = cv2.cvtColor(operators, cv2.COLOR_RGB2GRAY)

    plus = cv2.resize(~operators[:, :316], (28, 28)).reshape(1, 28, 28, 1)
    equal = cv2.resize(~operators[:, 340:340 + 316], (28, 28)).reshape(1, 28, 28, 1)
    minus = cv2.resize(~operators[:, 710:710 + 316], (28, 28)).reshape(1, 28, 28, 1)
    divide = cv2.resize(~operators[:, 1079:1079 + 316], (28, 28)).reshape(1, 28, 28, 1)
    multiply = cv2.resize(~operators[:, 1420:1420 + 316], (28, 28)).reshape(1, 28, 28, 1)

    op_datagen = ImageDataGenerator(
        rotation_range=360,
        zoom_range=[0.9, 1.6],
        vertical_flip=True,
        horizontal_flip=True)
    plus_numpy = np.array([list(op_datagen.flow(plus)) for _ in range(10000)])
    equal_numpy = np.array([list(op_datagen.flow(equal)) for _ in range(10000)])
    minus_numpy = np.array([list(op_datagen.flow(minus)) for _ in range(10000)])
    divide_numpy = np.array([list(op_datagen.flow(divide)) for _ in range(10000)])
    multiply_numpy = np.array([list(op_datagen.flow(multiply)) for _ in range(10000)])

    x_op = np.concatenate((plus_numpy, minus_numpy, multiply_numpy, divide_numpy, equal_numpy))
    y_op = np.concatenate((9 * np.ones(len(plus)), 10 * np.ones(len(minus)), 11 * np.ones(len(multiply)), 12 * np.ones(len(divide)), 13 * np.ones(len(equal))))

    rand_perm = np.random.permutation(len(y_op))
    x_op = x_op[rand_perm]
    y_op = y_op[rand_perm]

    x_op_train = x_op[:27000]
    x_op_test = x_op[27000:]
    y_op_train = y_op[:27000]
    y_op_test = y_op[27000:]

    # load digits and operators in the video
    cap = cv2.VideoCapture(vid_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    two = []
    three = []
    seven = []
    plus = []
    multiply = []
    divide = []
    equal = []

    angles = [0, 45, 100, 185, 228, 274, 350]
    for ang in angles:
        M = cv2.getRotationMatrix2D((360, 240), ang, 1.0)
        tmp = cv2.warpAffine(frames[0], M, (720, 480))
        tip, center = find_red_arrow(tmp)
        elements, _ = find_math_elements(tmp, center, 18)
        plus.append(255. - normalize(elements[0]))
        three.append(255. - normalize(elements[1]))
        two.append(255. - normalize(elements[2]))
        divide.append(255. - normalize(elements[3]))
        seven.append(255. - normalize(elements[4]))
        seven.append(255. - normalize(elements[5]))
        equal.append(255. - normalize(elements[6]))
        multiply.append(255. - normalize(elements[7]))
        three.append(255. - normalize(elements[8]))
        two.append(255. - normalize(elements[9]))

    two_numpy = np.array([np.array(img) for img in two])
    three_numpy = np.array([np.array(img) for img in three])
    seven_numpy = np.array([np.array(img) for img in seven])
    plus_numpy = np.array([np.array(img) for img in plus])
    multiply_numpy = np.array([np.array(img) for img in multiply])
    divide_numpy = np.array([np.array(img) for img in divide])
    equal_numpy = np.array([np.array(img) for img in equal])

    x_vid = np.concatenate((two_numpy, three_numpy, seven_numpy, plus_numpy, multiply_numpy, divide_numpy, equal_numpy))
    y_vid = np.concatenate((2 * np.ones(len(two)), 3 * np.ones(len(three)), 7 * np.ones(len(seven)), 9 * np.ones(len(plus)), 11 * np.ones(len(multiply)), 12 * np.ones(len(divide)), 13 * np.ones(len(equal))))

    rand_perm = np.random.permutation(len(y_vid))
    x_vid = x_vid[rand_perm]
    y_vid = y_vid[rand_perm]

    x_vid_train = x_vid[:50]
    x_vid_test = x_vid[50:]
    y_vid_train = y_vid[:50]
    y_vid_test = y_vid[50:]

    x_train = np.concatenate((x_train, x_op_train, x_vid_train))
    y_train = np.concatenate((y_train, y_op_train, y_vid_train))
    x_test = np.concatenate((x_test, x_op_test, x_vid_test))
    y_test = np.concatenate((y_test, y_op_test, y_vid_test))

    return (x_train, y_train), (x_test, y_test)


def main():
    """ Main function """
    parser = argparse.ArgumentParser(description='Train the digits/operators classifier.')
    parser.add_argument('--op_path', type=str, help='Path to the operator image.')
    parser.add_argument('--vid_path', type=str, help='Path to the video.')
    parser.add_argument('--model', type=str, default='models/model.h5', help='Path to save the model.')

    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = load_data(args.op_path, args.vid_path)

    now = datetime.datetime.now
    num_classes = 14
    img_rows, img_cols = 28, 28
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)

    def train_model(model, train, test, num_classes, epochs, batch_size, steps):
        x_train = train[0].reshape((train[0].shape[0],) + input_shape)
        x_test = test[0].reshape((test[0].shape[0],) + input_shape)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(train[1], num_classes)
        y_test = keras.utils.to_categorical(test[1], num_classes)

        model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(),
                    metrics=['accuracy'])

        t = now()

        datagen = ImageDataGenerator(
            rotation_range=360,
            zoom_range = [0.9, 1.6],
            width_shift_range=0.1,
            height_shift_range=0.1)


        #datagen.fit(X_train)
        train_gen = datagen.flow(x_train, y_train, batch_size=batch_size)
        test_gen = datagen.flow(x_test, y_test, batch_size=batch_size)

        history = model.fit_generator(train_gen,
                                    epochs=epochs,
                                    steps_per_epoch=steps,
                                    validation_data=test_gen,
                                    validation_steps=steps
        )

        print('Training time: %s' % (now() - t))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

    # define two groups of layers: feature (convolutions) and classification (dense)
    feature_layers = [
        Conv2D(16, 3, padding='same', input_shape=input_shape),
        Activation('relu'),
        Conv2D(16, 3, padding='same'),
        Activation('relu'),
        MaxPooling2D(pool_size=2),
        Dropout(0.25),
        Conv2D(32, 3, padding='same'),
        Activation('relu'),
        Conv2D(32, 3, padding='same'),
        Activation('relu'),
        Dropout(0.25),
        Flatten(),
    ]

    classification_layers = [
        Dense(256),
        Activation('relu'),
        Dropout(0.25),
        Dense(num_classes),
        Activation('softmax')
    ]

    # create complete model
    model = Sequential(feature_layers + classification_layers)

    # train model
    train_model(model,
                (x_train, y_train),
                (x_test, y_test), num_classes, 30, 256, 500)

    os.makedirs(os.path.split(args.model)[0], exist_ok=True)
    model.save(args.model)


if __name__ == '__main__':

    main()
