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

import tensorflow_datasets as tfds
import tensorflow as tf
import random
import os

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

def data_aug_img_layer(images, seed_):

    from tensorflow.keras.layers import Lambda

    def aug_img(img_batch, seed_):
        def random_flip_preproc(img_batch, seed_):
            img_batch = tf.image.random_flip_left_right(img_batch, seed=seed_)
            return img_batch

        def random_crop_preproc(img_batch, seed_):

            original_shape = ors = list(img_batch.shape)

            # Percentage of pixels to crop
            crp_p = random.sample([4, 8, 16], 1)[0]

            crop_size = [
                ors[0],
                ors[1] - int(ors[1] / crp_p),
                ors[2] - int(ors[2] / crp_p),
                ors[3],
            ]

            img_batch = tf.image.random_crop(img_batch, crop_size, seed=seed_)
            img_batch = tf.image.resize(
                img_batch, original_shape[1:3], method="nearest"
            )

            return img_batch

        choice = random.sample([0, 1, 2, 3, 4], 1)[0]

        if choice == 0:
            img_batch = random_flip_preproc(img_batch, seed_)
        elif choice == 1:
            img_batch = random_crop_preproc(img_batch, seed_)
        elif choice == 2:
            img_batch = random_flip_preproc(img_batch, seed_)
            img_batch = random_crop_preproc(img_batch, seed_)
        elif choice == 3:
            img_batch = random_crop_preproc(img_batch, seed_)
            img_batch = random_flip_preproc(img_batch, seed_)
        else:
            pass

        return img_batch

    # Keras Lambda layer
    return Lambda(lambda x: aug_img(x, seed_))(images)

def custom_dataset(
    name,
    train_classes,
    test_classes,
    tr_imgs_per_class,
    ts_imgs_per_class,
    seed,
    shuffle=False,
    shuffle_buffer_size=60000,
):
    assert name in ["mnist", "cifar10", "cifar100"], "choose: mnist, cifar10, cifar100"

    num_cl = {"mnist": 10, "cifar10": 10, "cifar100": 100}

    for class_tr, class_ts in zip(train_classes, test_classes):
        assert class_tr in range(num_cl[name]) and class_ts in range(num_cl[name]), (
            "choice should be within (0-" + str(num_cl[name]) + ")"
        )

    assert tr_imgs_per_class != 0, "choices !=0 or -1 for all"
    assert tr_imgs_per_class >= -1, "choices !=0 or -1 for all"

    assert ts_imgs_per_class != 0, "choices !=0 or -1 for all"
    assert ts_imgs_per_class >= -1, "choices !=0 or -1 for all"

    print("[Acquire dataset and preproc]\n")

    (train_ds, test_ds), info = tfds.load(
        name, split=["train", "test"], with_info=True, as_supervised=True
    )

    all_classes = [
        info.features["label"].str2int(st) for st in info.features["label"].names
    ]

    original_im_shape = info.features["image"].shape

    print("[Choose images by label name]:" + str(train_classes))

    train_ds_part = train_ds
    test_ds_part = test_ds

    if train_classes != []:
        train_ds_part = train_ds.filter(lambda x, y: y == train_classes[0]).take(
            tr_imgs_per_class
        )
        for cl in train_classes[1:]:
            train_ds_part = train_ds_part.concatenate(
                train_ds.filter(lambda x, y: y == cl).take(tr_imgs_per_class)
            )

    if test_classes != []:
        test_ds_part = test_ds.filter(lambda x, y: y == test_classes[0]).take(
            ts_imgs_per_class
        )
        for cl in test_classes[1:]:
            test_ds_part = test_ds_part.concatenate(
                test_ds.filter(lambda x, y: y == cl).take(ts_imgs_per_class)
            )

    def proc(img, label):
        img = tf.cast(img, dtype=tf.float32)
        img = img / 255.0
        label = tf.one_hot(label, depth=len(all_classes))
        return img, label

    if train_classes != []:
        train_ds_part = train_ds_part.map(proc)

    if test_classes != []:
        test_ds_part = test_ds_part.map(proc)

    if shuffle:
        train_ds_part = train_ds_part.shuffle(shuffle_buffer_size, seed=seed)

    return train_ds_part, test_ds_part, all_classes, original_im_shape


def dataset_training(model, dataset, epochs_, opt, loss, seed_, shuf=False):

    train = tf.function(train_step_initiate_graph_function())
    dataset = dataset.shuffle(60000, seed=seed_).cache()

    for i in range(1, epochs_ + 1):
        print("Epoch:" + str(i) + "|" + str(epochs_), end="\r")
        for j, (images, labels) in enumerate(dataset):
            images = data_aug_img_layer(images, seed_ + i + j)
            train(model, images, labels, loss, opt)

