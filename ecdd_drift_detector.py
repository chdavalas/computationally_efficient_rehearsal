#!/usr/bin/env python

# Copyright 2022 chdavalas
#
# chdavalas@gmail.com, cdavalas@hua.gr
#
# This program is free software; you can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program;
# if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301, USA.
#

import numpy as np
import tensorflow as tf
from numpy import argmax
from math import sqrt, log, ceil, log2

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CUDNN_DETERMINISTIC"] = "true"
os.environ["TF_DETERMINISTIC_OPS"] = "1"


class ECDDetector:
    def __init__(self, lam=0.2, avg_run_len=400):
        self._lam = lam
        self._avg_run_len = avg_run_len
        self._t = 0
        self._p_0t = 0
        self._Z_t = 0
        self._X_t = 0
        self._diff_X_t = 0
        self._drift = False

    def _L_t(self):
        if self._avg_run_len == 100:
            return (
                2.76
                - 6.23 * self._p_0t
                + 18.12 * pow(self._p_0t, 3)
                - 312.45 * pow(self._p_0t, 5)
                + 1002.18 * pow(self._p_0t, 7)
            )
        elif self._avg_run_len == 400:
            return (
                3.97
                - 6.56 * self._p_0t
                + 48.73 * pow(self._p_0t, 3)
                - 330.13 * pow(self._p_0t, 5)
                + 848.18 * pow(self._p_0t, 7)
            )
        else:
            return (
                1.17
                + 7.56 * self._p_0t
                - 21.24 * pow(self._p_0t, 3)
                + 112.12 * pow(self._p_0t, 5)
                - 987.24 * pow(self._p_0t, 7)
            )

    def predict(self, model, images, labels):
        predicted_labels = argmax(tf.nn.softmax(model(images), axis=-1), axis=-1)
        real_labels = argmax(labels, axis=1)
        n = predicted_labels.shape[0]
        X_t = np.sum(predicted_labels != real_labels)
        self._diff_X_t = X_t / n - self._X_t
        self._X_t = X_t / n

        # X_t is sum, so second term is average
        self._p_0t = (self._t / (self._t + n)) * self._p_0t + (1 / (self._t + n)) * X_t
        self._t += n
        sxt = self._p_0t * (1 - self._p_0t)
        szt = (
            sqrt((self._lam / (2 - self._lam)) * (1 - (1 - self._lam) ** (2 * self._t)))
            * sxt
        )
        L_t = self._L_t()
        self._Z_t = (1 - self._lam) * self._Z_t + self._lam * (X_t / n)
        self._drift = (self._Z_t > self._p_0t + L_t * szt)
        return self._drift

    @property
    def X_t(self):
        return self._X_t

    @property
    def diff_X_t(self):
        return self._diff_X_t

    @property
    def Z_t(self):
        return self._Z_t

    @property
    def p_0t(self):
        return self._p_0t

    @property
    def drift(self):
        return self._drift

    @property
    def time(self):
        return self._t
