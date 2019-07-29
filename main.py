import copy
import gym
import os
import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime

MAX_EXPERIENCES = 500000
MIN_EXPERIENCES = 50000
TARGET_UPDATE_PERIOD = 10000
IM_SIZE = 84
K = 4

class ImageTransformer:
    def __init__(self):
        with tf.variable_scope('image_transformer'):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(self.output, [IM_SIZE, IM_SIZE], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def transform(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.output, { self.input_state: state })

def update_state(state, obs_small):
    return np.append(state[:, :, 1:], np.expand_dims(obs_small, 2), axis=2)

class ReplayMemory:
    def __init__(self):
        pass