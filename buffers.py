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

from time import time, strftime, sleep
from tensorflow import keras
import numpy as np
import tensorflow as tf
from numpy import argmax

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CUDNN_DETERMINISTIC"] = "true"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

class RehearsalBuffer:
    def __init__(self, model, images, labels):
        self._model = model
        self._images = images
        self._labels = labels

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def update(self, model, images, labels):
        pass


class DynamicRehearsalBuffer(RehearsalBuffer):
    def __init__(self, model, images, labels):
        super().__init__(model, images, labels)
        self._class_avgs = {}
        self._num_per_class = {}
        self._ind_per_class = {}
        self._thetas_per_class = {}
        self._thetas = self._compute_logits(images)

        for i, (theta, label) in enumerate(zip(self._thetas, self._labels)):
            lbl = argmax(label)
            if lbl not in self._class_avgs:
                self._class_avgs[lbl] = theta
                self._num_per_class[lbl] = 1
                self._ind_per_class[lbl] = [i]
                self._thetas_per_class[lbl] = [theta]
            else:
                self._class_avgs[lbl] += theta
                self._num_per_class[lbl] += 1
                self._ind_per_class[lbl].append(i)
                self._thetas_per_class[lbl].append(theta)

        for key in self._class_avgs.keys():
            self._class_avgs[key] = tf.scalar_mul(
                1 / self._num_per_class[key], self._class_avgs[key]
            )

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def _norm(self, x, y):
        return tf.tensordot(x - y, x - y, axis=(0, 0))

    def update(self, model, new_images, new_labels):
        k = time()
        new_thetas = self._compute_logits(new_images)

        for theta_xi, nimg, nlbl in zip(new_thetas, new_images, new_labels):

            yi = argmax(nlbl)
            nyi = self._num_per_class[yi]

            self._class_avgs[yi] *= nyi / (nyi + 1)
            self._class_avgs[yi] += (1 / (nyi + 1)) * theta_xi

            max_dist = None
            max_dist_index = None

            class_avgs_yi = self._class_avgs[yi]
            class_thetas = self._thetas_per_class[yi]

            fn = lambda x: tf.math.reduce_euclidean_norm(tf.stack([x, class_avgs_yi]))
            dists = tf.convert_to_tensor([fn(x) for x in class_thetas])

            max_dist_index_in_class = tf.math.argmax(dists).numpy()
            max_dist = dists[max_dist_index_in_class]
            max_dist_index = self._ind_per_class[yi][max_dist_index_in_class]

            d_nimg = tf.math.reduce_euclidean_norm(tf.stack([theta_xi, class_avgs_yi]))

            if d_nimg <= max_dist:
                self._images = self._replace_in_tensor(
                    self._images, nimg, max_dist_index
                )
                self._thetas = self._replace_in_tensor(
                    self._thetas, theta_xi, max_dist_index
                )
                self._labels = self._replace_in_tensor(
                    self._labels, nlbl, max_dist_index
                )
                self._thetas_per_class[yi][max_dist_index_in_class] = theta_xi

        l = time()
        print("buffer update time: {}".format(l - k))

    def _replace_in_tensor(self, tensor, element, index_):
        # print("Replacing item {} in tensor with length {}".format(index_, len(tensor)))

        if index_ == 0:
            return tf.concat(
                [
                    tf.convert_to_tensor([element]),
                    tf.gather(tensor, list(range(1, len(tensor))), axis=0),
                ],
                axis=0,
            )
        elif index_ == len(tensor) - 1:
            return tf.concat(
                [
                    tf.gather(tensor, list(range(len(tensor) - 1)), axis=0),
                    tf.convert_to_tensor([element]),
                ],
                axis=0,
            )
        else:
            return tf.concat(
                [
                    tf.gather(tensor, list(range(index_)), axis=0),
                    tf.convert_to_tensor([element]),
                    tf.gather(tensor, list(range(index_ + 1, len(tensor))), axis=0),
                ],
                axis=0,
            )

    def _compute_logits(self, images, batch_size=256):
        result = None
        for batch in tf.data.Dataset.from_tensor_slices(images).batch(
            batch_size=batch_size
        ):
            logits = self._model(batch)
            if result is None:
                result = logits
            else:
                result = tf.concat([result, logits], axis=0)
        return result

def randomized_buffer_refresh(
    buffer_im, buffer_la, new_im, new_la, cut_size, buf_lim, seed_
):
    CURRENT_BUFFER_SIZE = len(buffer_im.numpy())

    # MAKE USER THE SAMPLE REQUESTED RESPECTS BATCH SIZE
    sample_len = min(len(new_im.numpy()), cut_size)

    # TAKE SAMPLE FROM NEW BATCH
    ind = random.sample(range(len(new_im.numpy())), sample_len)
    new_im = tf.gather(new_im, ind, axis=0)
    new_la = tf.gather(new_la, ind, axis=0)

    # IF BUFFER IS FULL REPLACE OLD ELEMENTS FROM BUFFER WITH NEW ELEMENTS
    # (SAMPLE) FROM BATCH
    if CURRENT_BUFFER_SIZE >= buf_lim:

        ind = random.sample(
            range(CURRENT_BUFFER_SIZE), CURRENT_BUFFER_SIZE - sample_len
        )
        buffer_im = tf.gather(buffer_im, ind, axis=0)
        buffer_la = tf.gather(buffer_la, ind, axis=0)

    # ADD NEW ELEMENTS
    buffer_im = tf.concat([buffer_im, new_im], axis=0)
    buffer_la = tf.concat([buffer_la, new_la], axis=0)

    # SHUFFLE BUFFER
    ind = random.sample(range(len(buffer_im.numpy())), len(buffer_im.numpy()))
    buffer_im = tf.gather(buffer_im, ind, axis=0)[:buf_lim]
    buffer_la = tf.gather(buffer_la, ind, axis=0)[:buf_lim]

    return buffer_im, buffer_la

def mix_batches(
    bufr_im,
    bufr_la,
    bufr_index_start,
    bufr_mix_batch,
    post_im,
    post_la,
    post_index_start,
    post_mix_batch,
    total_batches=None,
    seed=32221100,
):
    batch = bufr_mix_batch + post_mix_batch
    if batch <= 0:
        raise ValueError("Illegal batch size requested: {}".format(batch))

    bufr_len = len(bufr_im)
    post_len = len(post_im)
    if total_batches is None:
        total_batches = min(bufr_len // bufr_mix_batch, post_len // post_mix_batch)

    bufr_cur = bufr_index_start
    post_cur = post_index_start
    batches_cur = 0

    im_mix_batches = []
    la_mix_batches = []

    while batches_cur < total_batches:
        bufr_indxs = tf.constant(
            [v % bufr_len for v in range(bufr_cur, bufr_cur + bufr_mix_batch)]
        )
        bufr_cur = (bufr_cur + bufr_mix_batch) % bufr_len
        bufr_im_mix = tf.gather(bufr_im, bufr_indxs)
        bufr_la_mix = tf.gather(bufr_la, bufr_indxs)

        post_indxs = tf.constant(
            [v % post_len for v in range(post_cur, post_cur + post_mix_batch)]
        )
        post_cur = (post_cur + post_mix_batch) % post_len
        post_im_mix = tf.gather(post_im, post_indxs)
        post_la_mix = tf.gather(post_la, post_indxs)

        # print("Mixed batch postponed {},{} rehearsal".format(post_indxs, bufr_indxs))

        im_mix_batches += [tf.concat([bufr_im_mix, post_im_mix], axis=0)]
        la_mix_batches += [tf.concat([bufr_la_mix, post_la_mix], axis=0)]

        batches_cur += 1

    return im_mix_batches, la_mix_batches, bufr_cur, post_cur

