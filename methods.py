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

from collections import defaultdict
import pandas as pd
import tensorflow_datasets as tfds
import random
from datetime import datetime
from time import time, strftime, sleep
from sys import setrecursionlimit
import argparse
from tensorflow import keras
import numpy as np
import tensorflow as tf
from numpy import argmax
from math import sqrt, log, ceil, log2
from statistics import mean

from datasets import custom_dataset, dataset_training
from models import ResNet32
from buffers import RehearsalBuffer, DynamicRehearsalBuffer 
from buffers import randomized_buffer_refresh, mix_batches
from ecdd_drift_detector import ECDDetector


import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CUDNN_DETERMINISTIC"] = "true"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

def train_step_initiate_graph_function():
    def train_step(model, images, labels, loss, opt):

        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            soft_logits = tf.nn.softmax(logits)
            loss_value = loss(labels, soft_logits) + sum(model.losses)

        grads = tape.gradient(loss_value, model.trainable_weights)
        opt.apply_gradients(zip(grads, model.trainable_weights))

        return loss_value

    return train_step

class OnlineTraining:
    def __init__(
        self,
        name,
        model,
        optimizer,
        loss,
        scoreboard,
        repeat,
        seed,
        augment_images=False,
    ):
        self._name = name
        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self._cur_batch = 0
        self._scoreboard = scoreboard
        self._repeat = repeat
        self._seed = seed
        self._augment_images = augment_images
        self._online_accuracy = tf.keras.metrics.Accuracy()
        self._train = tf.function(train_step_initiate_graph_function())

    @property
    def model(self):
        return self._model

    @property
    def name(self):
        return self._name

    def update(self, images, labels):
        print("\n####################:")
        print("[{}][{}]:".format(self._name, self._cur_batch))

        self._online_accuracy.update_state(
            argmax(tf.nn.softmax(self._model(images), axis=-1), axis=-1),
            argmax(labels, axis=1),
        )
        self._scoreboard["online_acc_{}".format(self._name)] += [
            self._online_accuracy.result().numpy()
        ]
        print("Online accuracy: {}".format(self._online_accuracy.result().numpy()))

        start = time()

        print("Training {} times".format(self._repeat))

        self._optimizer.iterations.assign(0)
        self._optimizer.learning_rate.decay_steps = 1

        train_batches = 0
        for i in range(self._repeat):
            train_images = images
            if self._augment_images: 
                train_images = data_aug_img_layer(images, self._seed + i + 1)
            self._train(self._model, train_images, labels, self._loss, self._optimizer)
            train_batches += 1
            sys.stdout.flush()

        self._cur_batch += 1
        stop = time()
        self._scoreboard["repeat_{}".format(self._name)] += [self._repeat]
        self._scoreboard["times_{}".format(self._name)] += [stop - start]
        self._scoreboard["train_batches_{}".format(self._name)] += [train_batches]


class ContinuousRehearsal:
    def __init__(
        self,
        name,
        model,
        optimizer,
        loss,
        rehearsal_buffer_images,
        rehearsal_buffer_labels,
        rehearsal_repeats,
        train_every_steps,
        scoreboard,
        seed,
        mix_len,
        augment_images=False,
    ):
        self._name = name
        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self._cur_batch = 0
        self._trained_on_last_batch = False
        self._postponed_images = None
        self._postponed_labels = None
        self._postponed_buffer_index = 0
        self._rehearsal_buffer = DynamicRehearsalBuffer(
            model, rehearsal_buffer_images, rehearsal_buffer_labels
        )
        self._rehearsal_buffer_index = 0
        self._rehearsal_repeats = rehearsal_repeats
        self._train_every_steps = train_every_steps
        self._scoreboard = scoreboard
        self._seed = seed
        self._mix_len = mix_len
        self._augment_images = augment_images
        self._online_accuracy = tf.keras.metrics.Accuracy()
        self._train = tf.function(train_step_initiate_graph_function())

    @property
    def model(self):
        return self._model

    @property
    def name(self):
        return self._name

    def update(self, images, labels):
        print("\n####################:")
        print("[{}][{}]:".format(self._name, self._cur_batch))

        self._online_accuracy.update_state(
            argmax(tf.nn.softmax(self._model(images), axis=-1), axis=-1),
            argmax(labels, axis=1),
        )
        self._scoreboard["online_acc_{}".format(self._name)] += [
            self._online_accuracy.result().numpy()
        ]
        print("Online accuracy: {}".format(self._online_accuracy.result().numpy()))

        start = time()

        if self._cur_batch == 0 or self._trained_on_last_batch:
            self._postponed_images = images
            self._postponed_labels = labels
        else:
            self._postponed_images = tf.concat([self._postponed_images, images], axis=0)
            self._postponed_labels = tf.concat([self._postponed_labels, labels], axis=0)

        train_batches = 0
        rehearsal_repeats = 0

        if self._cur_batch % self._train_every_steps == 0:
            rehearsal_repeats = self._rehearsal_repeats

            print("Training {} times".format(rehearsal_repeats))

            self._optimizer.iterations.assign(0)
            batches_per_rehearsal_repeat = len(self._postponed_images) // self._mix_len
            self._optimizer.learning_rate.decay_steps = batches_per_rehearsal_repeat

            for _ in range(self._rehearsal_repeats):
                (
                    mix_im_list,
                    mix_la_list,
                    self._rehearsal_buffer_index,
                    self._postponed_buffer_index,
                ) = mix_batches(
                    self._rehearsal_buffer.images,
                    self._rehearsal_buffer.labels,
                    self._rehearsal_buffer_index,
                    self._mix_len,
                    self._postponed_images,
                    self._postponed_labels,
                    self._postponed_buffer_index,
                    self._mix_len,
                    total_batches=batches_per_rehearsal_repeat,
                    seed=self._seed,
                )

                for mix_im, mix_la in zip(mix_im_list, mix_la_list):
                    train_images = mix_im
                    if self._augment_images: 
                        train_images = data_aug_img_layer(mix_im, self._seed)
                    self._train(
                        self._model, train_images, mix_la, self._loss, self._optimizer
                    )
                    train_batches += 1
                    sys.stdout.flush()
            self._trained_on_last_batch = True
        else:
            self._trained_on_last_batch = False

        self._rehearsal_buffer.update(self._model, images, labels)
        self._cur_batch += 1
        stop = time()
        self._scoreboard["repeat_{}".format(self._name)] += [rehearsal_repeats]
        self._scoreboard["times_{}".format(self._name)] += [stop - start]
        self._scoreboard["train_batches_{}".format(self._name)] += [train_batches]


class ContinuousRehearsalConverge:
    def __init__(
        self,
        name,
        model,
        optimizer,
        loss,
        rehearsal_buffer_images,
        rehearsal_buffer_labels,
        train_every_steps,
        scoreboard,
        seed,
        mix_len,
        augment_images=False,
        alpha_short=0.5,
        alpha_long=0.05,
        eps=0.005,
    ):
        self._name = name
        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self._cur_batch = 0
        self._trained_on_last_batch = False
        self._postponed_images = None
        self._postponed_labels = None
        self._postponed_buffer_index = 0
        self._rehearsal_buffer = DynamicRehearsalBuffer(
            model, rehearsal_buffer_images, rehearsal_buffer_labels
        )
        self._rehearsal_buffer_index = 0
        self._train_every_steps = train_every_steps
        self._scoreboard = scoreboard
        self._seed = seed
        self._mix_len = mix_len
        self._augment_images = augment_images
        self._loss_alpha_short = alpha_short
        self._loss_alpha_long = alpha_long
        self._eps = eps
        # TODO: Fixed value for now. Explore option of "warmup" for 5
        # iterations and then monitor difference.
        self._running_avg_loss_short = 1.0
        self._running_avg_loss_long = 1.0
        self._online_accuracy = tf.keras.metrics.Accuracy()
        self._train = tf.function(train_step_initiate_graph_function())

    @property
    def model(self):
        return self._model

    @property
    def name(self):
        return self._name

    def update(self, images, labels):
        print("\n####################:")
        print("[{}][{}]:".format(self._name, self._cur_batch))

        self._online_accuracy.update_state(
            argmax(tf.nn.softmax(self._model(images), axis=-1), axis=-1),
            argmax(labels, axis=1),
        )
        self._scoreboard["online_acc_{}".format(self._name)] += [
            self._online_accuracy.result().numpy()
        ]
        print("Online accuracy: {}".format(self._online_accuracy.result().numpy()))

        start = time()

        if self._cur_batch == 0 or self._trained_on_last_batch:
            self._postponed_images = images
            self._postponed_labels = labels
        else:
            self._postponed_images = tf.concat([self._postponed_images, images], axis=0)
            self._postponed_labels = tf.concat([self._postponed_labels, labels], axis=0)

        train_batches = 0
        rehearsal_repeats = 0
        if self._cur_batch % self._train_every_steps == 0:
            print("Starting training until convergence with eps={}".format(self._eps))

            stop_training = False
            self._optimizer.iterations.assign(0)
            batches_per_rehearsal_repeat = len(self._postponed_images) // self._mix_len
            self._optimizer.learning_rate.decay_steps = batches_per_rehearsal_repeat
            while not stop_training:
                (
                    mix_im_list,
                    mix_la_list,
                    self._rehearsal_buffer_index,
                    self._postponed_buffer_index,
                ) = mix_batches(
                    self._rehearsal_buffer.images,
                    self._rehearsal_buffer.labels,
                    self._rehearsal_buffer_index,
                    self._mix_len,
                    self._postponed_images,
                    self._postponed_labels,
                    self._postponed_buffer_index,
                    self._mix_len,
                    seed=self._seed,
                )

                # Avoid huge variable names
                l = self._running_avg_loss_short
                ll = self._running_avg_loss_long
                a = self._loss_alpha_short
                al = self._loss_alpha_long
                for mix_im, mix_la in zip(mix_im_list, mix_la_list):
                    train_images = mix_im
                    if self._augment_images: 
                        train_images = data_aug_img_layer(mix_im, self._seed)
                    loss = self._train(
                        self._model, train_images, mix_la, self._loss, self._optimizer
                    )
                    sys.stdout.flush()
                    train_batches += 1
                    l = (1 - a) * l + a * loss
                    ll = (1 - al) * ll + al * loss
                    # print('Loss: {}, '
                    #       'Running Avg (short): {}, '
                    #       'Running Avg (long): {}, '
                    #       'Abs. Diff: {}'.
                    #       format(loss, l, ll, abs(ll - l)))
                    if abs(ll - l) < self._eps:
                        stop_training = True
                rehearsal_repeats += 1
                self._running_avg_loss_short = l
                self._running_avg_loss_long = ll
                self._trained_on_last_batch = True
                if stop_training:
                    print(
                        "Stopping training after {} batches and {} repeats".format(
                            train_batches, rehearsal_repeats
                        )
                    )
                    sys.stdout.flush()

        else:
            self._trained_on_last_batch = False

        self._rehearsal_buffer.update(self._model, images, labels)
        self._cur_batch += 1
        stop = time()
        self._scoreboard["repeat_{}".format(self._name)] += [rehearsal_repeats]
        self._scoreboard["times_{}".format(self._name)] += [stop - start]
        self._scoreboard["train_batches_{}".format(self._name)] += [train_batches]


class DriftActivatedRehearsal:
    def __init__(
        self,
        name,
        model,
        optimizer,
        loss,
        rehearsal_buffer_images,
        rehearsal_buffer_labels,
        rehearsal_repeats,
        scoreboard,
        seed,
        mix_len,
        err_thr,
        max_notrain,
        use_rehearsal_drift_detector=False,
        rehearsal_drift_detector_batch=10,
        augment_images=False,
        dynamic_initial_learning_rate=True,
        dynamic_rehearsal_repeats=True,
        avg_run_len=100
    ):
        self._name = name
        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self._cur_batch = 0
        self._drift_detector = ECDDetector(avg_run_len=avg_run_len)
        self._postponed_images = None
        self._postponed_labels = None
        self._postponed_buffer_index = 0
        self._rehearsal_buffer = DynamicRehearsalBuffer(
            model, rehearsal_buffer_images, rehearsal_buffer_labels
        )
        self._rehearsal_buffer_index = 0
        self._rehearsal_repeats = rehearsal_repeats
        self._scoreboard = scoreboard
        self._seed = seed
        self._mix_len = mix_len
        self._err_thr = err_thr
        self._max_notrain = max_notrain
        self._last_train = 0
        self._rehearsal_drift_detector = (
            ECDDetector(avg_run_len=avg_run_len) if use_rehearsal_drift_detector else None
        )
        self._rehearsal_drift_detector_batch = rehearsal_drift_detector_batch
        self._augment_images = augment_images

        # Store initial learning rate from optimizer
        if callable(self._optimizer.learning_rate):
            self._initial_learning_rate = self._optimizer.learning_rate(0).numpy()
        else:
            self._initial_learning_rate = self._optimizer.learning_rate.numpy()

        self._dynamic_initial_learning_rate = dynamic_initial_learning_rate
        self._dynamic_rehearsal_repeats = dynamic_rehearsal_repeats

        self._online_accuracy = tf.keras.metrics.Accuracy()
        self._train = tf.function(train_step_initiate_graph_function())

    @property
    def model(self):
        return self._model

    @property
    def name(self):
        return self._name

    def _compute_training_repeat(self, detected_drift, Z_t):
        if self._dynamic_rehearsal_repeats:
            return ceil(2 * self._rehearsal_repeats * log2(1 + Z_t))
        else:
            return self._rehearsal_repeats

    def _compute_initial_learning_rate(self, detected_drift, Z_t):
        if self._dynamic_initial_learning_rate:
            return self._initial_learning_rate * min(100, 5 * np.exp(3 * Z_t))
        else:
            return self._initial_learning_rate

    def update(self, images, labels):
        print("\n####################:")
        print("[{}][{}]:".format(self._name, self._cur_batch))

        self._online_accuracy.update_state(
            argmax(tf.nn.softmax(self._model(images), axis=-1), axis=-1),
            argmax(labels, axis=1),
        )
        self._scoreboard["online_acc_{}".format(self._name)] += [
            self._online_accuracy.result().numpy()
        ]
        print("Online accuracy: {}".format(self._online_accuracy.result().numpy()))

        start = time()

        if self._cur_batch == 0 or self._cur_batch == self._last_train + 1:
            self._postponed_images = images
            self._postponed_labels = labels
        else:
            self._postponed_images = tf.concat([self._postponed_images, images], axis=0)
            self._postponed_labels = tf.concat([self._postponed_labels, labels], axis=0)

        detected_drift = self._drift_detector.predict(self._model, images, labels)
        actual_Z_t = self._drift_detector.Z_t
        actual_p_0t = self._drift_detector.p_0t
        actual_diff_X_t = self._drift_detector.diff_X_t
        actual_X_t = self._drift_detector.X_t

        if self._rehearsal_drift_detector is not None:
            rehearsal_sample_idxs = random.sample(
                range(tf.shape(self._rehearsal_buffer.images)[0]),
                self._rehearsal_drift_detector_batch,
            )
            rehearsal_sample_idxs = tf.constant(rehearsal_sample_idxs)
            rehearsal_detected_drift = self._rehearsal_drift_detector.predict(
                self._model,
                tf.gather(self._rehearsal_buffer.images, rehearsal_sample_idxs),
                tf.gather(self._rehearsal_buffer.labels, rehearsal_sample_idxs),
            )
            if rehearsal_detected_drift:
                detected_drift = True
                actual_Z_t = max(actual_Z_t, self._rehearsal_drift_detector.Z_t)
                actual_p_0t = max(actual_p_0t, self._rehearsal_drift_detector.p_0t)
                actual_diff_X_t = min(
                    actual_diff_X_t, self._rehearsal_drift_detector.diff_X_t
                )
                actual_X_t = max(actual_X_t, self._rehearsal_drift_detector.X_t)

        print("detected_drift = {}".format(detected_drift))
        print("X_t = {}".format(actual_X_t))
        print("diff_X_t = {}".format(actual_diff_X_t))
        print("Z_t = {}".format(actual_Z_t))
        print("p_0t = {}".format(actual_p_0t))
        sys.stdout.flush()
        self._scoreboard["detected_drift_{}".format(self._name)] += [detected_drift]
        self._scoreboard["diff_X_t_{}".format(self._name)] += [actual_diff_X_t]
        self._scoreboard["Z_t_{}".format(self._name)] += [actual_Z_t]
        self._scoreboard["p_0t_{}".format(self._name)] += [actual_p_0t]

        train_batches = 0
        rehearsal_repeat = 0

        # Conditions to start training
        if (
            detected_drift
            or actual_Z_t > self._err_thr
            or (self._cur_batch - self._last_train) > self._max_notrain
            #or actual_diff_X_t < -self._err_thr
        ):
            rehearsal_repeat = self._compute_training_repeat(detected_drift, actual_Z_t)
            initial_learning_rate = self._compute_initial_learning_rate(
                detected_drift, actual_Z_t
            )

            batches_per_rehearsal_repeat = len(self._postponed_images) // self._mix_len
            self._optimizer.iterations.assign(0)
            self._optimizer.learning_rate.initial_learning_rate = initial_learning_rate
            self._optimizer.learning_rate.decay_steps = batches_per_rehearsal_repeat

            print("Training {} times".format(rehearsal_repeat))
            print(
                "Using {} as initial learning rate".format(
                    self._optimizer.learning_rate.initial_learning_rate
                )
            )

            for _ in range(rehearsal_repeat):
                (
                    mix_im_list,
                    mix_la_list,
                    self._rehearsal_buffer_index,
                    self._postponed_buffer_index,
                ) = mix_batches(
                    self._rehearsal_buffer.images,
                    self._rehearsal_buffer.labels,
                    self._rehearsal_buffer_index,
                    self._mix_len,
                    self._postponed_images,
                    self._postponed_labels,
                    self._postponed_buffer_index,
                    self._mix_len,
                    total_batches=batches_per_rehearsal_repeat,
                    seed=self._seed,
                )

                for mix_im, mix_la in zip(mix_im_list, mix_la_list):
                    train_images = mix_im
                    if self._augment_images: 
                        train_images = data_aug_img_layer(mix_im, self._seed)
                    self._train(
                        self._model, train_images, mix_la, self._loss, self._optimizer
                    )
                    train_batches += 1
                    sys.stdout.flush()
            self._last_train = self._cur_batch

        self._rehearsal_buffer.update(self._model, images, labels)
        self._cur_batch += 1
        stop = time()
        self._scoreboard["repeat_{}".format(self._name)] += [rehearsal_repeat]
        self._scoreboard["times_{}".format(self._name)] += [stop - start]
        self._scoreboard["train_batches_{}".format(self._name)] += [train_batches]


class DriftActivatedRehearsalConverge:
    def __init__(
        self,
        name,
        model,
        optimizer,
        loss,
        rehearsal_buffer_images,
        rehearsal_buffer_labels,
        scoreboard,
        seed,
        mix_len,
        err_thr,
        max_notrain,
        use_rehearsal_drift_detector=False,
        rehearsal_drift_detector_batch=10,
        augment_images=False,
        dynamic_initial_learning_rate=True,
        alpha_short=0.5,
        alpha_long=0.05,
        eps=0.005,
        avg_run_len=100
    ):
        self._name = name
        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self._cur_batch = 0
        self._drift_detector = ECDDetector(avg_run_len=avg_run_len)
        self._postponed_images = None
        self._postponed_labels = None
        self._postponed_buffer_index = 0
        self._rehearsal_buffer = DynamicRehearsalBuffer(
            model, rehearsal_buffer_images, rehearsal_buffer_labels
        )
        self._rehearsal_buffer_index = 0
        self._scoreboard = scoreboard
        self._seed = seed
        self._mix_len = mix_len
        self._err_thr = err_thr
        self._max_notrain = max_notrain
        self._last_train = 0
        self._rehearsal_drift_detector = (
            ECDDetector(avg_run_len=avg_run_len) if use_rehearsal_drift_detector else None
        )
        self._rehearsal_drift_detector_batch = rehearsal_drift_detector_batch
        self._augment_images = augment_images
        self._loss_alpha_short = alpha_short
        self._loss_alpha_long = alpha_long
        self._eps = eps
        # Fixed
        self._running_avg_loss_short = 1.0
        self._running_avg_loss_long = 1.0

        # Store initial learning rate from optimizer
        if callable(self._optimizer.learning_rate):
            self._initial_learning_rate = self._optimizer.learning_rate(0).numpy()
        else:
            self._initial_learning_rate = self._optimizer.learning_rate.numpy()

        self._dynamic_initial_learning_rate = dynamic_initial_learning_rate
        self._online_accuracy = tf.keras.metrics.Accuracy()
        self._train = tf.function(train_step_initiate_graph_function())

    @property
    def model(self):
        return self._model

    @property
    def name(self):
        return self._name

    def _compute_initial_learning_rate(self, detected_drift, Z_t):
        if self._dynamic_initial_learning_rate:
            return self._initial_learning_rate * min(100, 5 * np.exp(3 * Z_t))
        else:
            return self._initial_learning_rate

    def update(self, images, labels):
        print("\n####################:")
        print("[{}][{}]:".format(self._name, self._cur_batch))

        self._online_accuracy.update_state(
            argmax(tf.nn.softmax(self._model(images), axis=-1), axis=-1),
            argmax(labels, axis=1),
        )
        self._scoreboard["online_acc_{}".format(self._name)] += [
            self._online_accuracy.result().numpy()
        ]
        print("Online accuracy: {}".format(self._online_accuracy.result().numpy()))

        start = time()

        if self._cur_batch == 0 or self._cur_batch == self._last_train + 1:
            self._postponed_images = images
            self._postponed_labels = labels
        else:
            self._postponed_images = tf.concat([self._postponed_images, images], axis=0)
            self._postponed_labels = tf.concat([self._postponed_labels, labels], axis=0)

        detected_drift = self._drift_detector.predict(self._model, images, labels)
        actual_Z_t = self._drift_detector.Z_t
        actual_p_0t = self._drift_detector.p_0t
        actual_diff_X_t = self._drift_detector.diff_X_t
        actual_X_t = self._drift_detector.X_t

        if self._rehearsal_drift_detector is not None:
            rehearsal_sample_idxs = random.sample(
                range(tf.shape(self._rehearsal_buffer.images)[0]),
                self._rehearsal_drift_detector_batch,
            )
            rehearsal_sample_idxs = tf.constant(rehearsal_sample_idxs)
            rehearsal_detected_drift = self._rehearsal_drift_detector.predict(
                self._model,
                tf.gather(self._rehearsal_buffer.images, rehearsal_sample_idxs),
                tf.gather(self._rehearsal_buffer.labels, rehearsal_sample_idxs),
            )
            if rehearsal_detected_drift:
                detected_drift = True
                actual_Z_t = max(actual_Z_t, self._rehearsal_drift_detector.Z_t)
                actual_p_0t = max(actual_p_0t, self._rehearsal_drift_detector.p_0t)
                actual_diff_X_t = min(
                    actual_diff_X_t, self._rehearsal_drift_detector.diff_X_t
                )
                actual_X_t = max(actual_X_t, self._rehearsal_drift_detector.X_t)

        print("detected_drift = {}".format(detected_drift))
        print("X_t = {}".format(actual_X_t))
        print("diff_X_t = {}".format(actual_diff_X_t))
        print("Z_t = {}".format(actual_Z_t))
        print("p_0t = {}".format(actual_p_0t))
        sys.stdout.flush()
        self._scoreboard["detected_drift_{}".format(self._name)] += [detected_drift]
        self._scoreboard["diff_X_t_{}".format(self._name)] += [actual_diff_X_t]
        self._scoreboard["Z_t_{}".format(self._name)] += [actual_Z_t]
        self._scoreboard["p_0t_{}".format(self._name)] += [actual_p_0t]

        train_batches = 0
        rehearsal_repeats = 0
        if (
            detected_drift
            or actual_Z_t > self._err_thr
            or (self._cur_batch - self._last_train) > self._max_notrain
            #or actual_diff_X_t < -self._err_thr
        ):
            print("Starting training until convergence with eps={}".format(self._eps))
            stop_training = False
            initial_learning_rate = self._compute_initial_learning_rate(
                detected_drift, actual_Z_t
            )
            batches_per_rehearsal_repeat = len(self._postponed_images) // self._mix_len
            self._optimizer.iterations.assign(0)
            self._optimizer.learning_rate.initial_learning_rate = initial_learning_rate
            self._optimizer.learning_rate.decay_steps = batches_per_rehearsal_repeat

            while not stop_training:
                (
                    mix_im_list,
                    mix_la_list,
                    self._rehearsal_buffer_index,
                    self._postponed_buffer_index,
                ) = mix_batches(
                    self._rehearsal_buffer.images,
                    self._rehearsal_buffer.labels,
                    self._rehearsal_buffer_index,
                    self._mix_len,
                    self._postponed_images,
                    self._postponed_labels,
                    self._postponed_buffer_index,
                    self._mix_len,
                    total_batches=batches_per_rehearsal_repeat,
                    seed=self._seed,
                )

                # Avoid huge variable names
                l = self._running_avg_loss_short
                ll = self._running_avg_loss_long
                a = self._loss_alpha_short
                al = self._loss_alpha_long
                for mix_im, mix_la in zip(mix_im_list, mix_la_list):
                    train_images = mix_im
                    if self._augment_images: 
                        train_images = data_aug_img_layer(mix_im, self._seed)
                    loss = self._train(
                        self._model, train_images, mix_la, self._loss, self._optimizer
                    )
                    sys.stdout.flush()
                    train_batches += 1
                    l = (1 - a) * l + a * loss
                    ll = (1 - al) * ll + al * loss
                    # print('Loss: {}, '
                    #       'Running Avg (short): {}, '
                    #       'Running Avg (long): {}, '
                    #       'Abs. Diff: {}'.
                    #       format(loss, l, ll, abs(ll - l)))
                    if abs(ll - l) < self._eps:
                        stop_training = True

                rehearsal_repeats += 1
                self._running_avg_loss_short = l
                self._running_avg_loss_long = ll
                self._last_train = self._cur_batch
                if stop_training:
                    print(
                        "Stopping training after {} batches and {} repeats".format(
                            train_batches, rehearsal_repeats
                        )
                    )
                    sys.stdout.flush()

        self._rehearsal_buffer.update(self._model, images, labels)
        self._cur_batch += 1
        stop = time()

        # rehearsal_repeats will always be zero
        self._scoreboard["repeat_{}".format(self._name)] += [rehearsal_repeats]
        self._scoreboard["times_{}".format(self._name)] += [stop - start]
        self._scoreboard["train_batches_{}".format(self._name)] += [train_batches]
