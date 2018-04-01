"""
The driving_log records the images files for each operation
There are six columns:
A. The image captured by center camera, 360x160x3, RGB
B. The image captured by left camera
C. The image captured by right camera
D. The steering angle: [-1, 1] -> [-25, 25]
E. Throttle: [0, 1]
F. Brake: all zero
G. Speed: [0, 30]

# test sync
"""

import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# every sample is a [filename, measurement]
samples = []
delta = [0, 0.2, -0.2]  # center, left, right
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        measurement = float(line[3])
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            samples.append([filename, measurement + delta[i]])

delta_left = [0.4, 0.6, 0.0]
with open('../data/driving_log_left.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        measurement = float(line[3])
        for i in (0, 1):  # center and left
            source_path = line[i]
            filename = source_path.split('/')[-1]
            samples.append([filename, measurement + delta_left[i]])

delta_right = [-0.4, 0.0, -0.6]
with open('../data/driving_log_right.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        measurement = float(line[3])
        for i in (0, 2):  # center and right
            source_path = line[i]
            filename = source_path.split('/')[-1]
            samples.append([filename, measurement + delta_right[i]])

print('Before augmentation: %s' % len(samples))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    assert batch_size % 2 == 0
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        samples_shuffled = shuffle(samples)
        for offset in range(0, num_samples, int(batch_size / 2)):
            batch_samples = samples_shuffled[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '../data/IMG/' + batch_sample[0]
                image = cv2.imread(name)
                angle = float(batch_sample[1])
                images.append(image)
                angles.append(angle)
                # agument data by flipping
                images.append(cv2.flip(image, 1))
                angles.append(-1.0 * angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

