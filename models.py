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

from tensorflow import keras
import tensorflow as tf

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CUDNN_DETERMINISTIC"] = "true"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

def residual_block(n_filters, strd, last=False):
    def BLOCK(input_layer):

        shortcut = input_layer

        from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add
        from tensorflow.keras.regularizers import l2
        from tensorflow.keras.activations import relu, elu

        layer = Conv2D(
            n_filters,
            kernel_size=(3, 3),
            kernel_regularizer=l2(1e-5),
            padding="same",
            strides=(strd, strd),
        )(input_layer)

        layer = BatchNormalization()(layer)
        layer = Activation(relu)(layer)

        layer = Conv2D(
            n_filters, kernel_size=(3, 3), kernel_regularizer=l2(1e-5), padding="same"
        )(layer)
        layer = BatchNormalization()(layer)

        if strd != 1:

            projection_layer = Conv2D(
                n_filters,
                kernel_size=(1, 1),
                padding="same",
                kernel_regularizer=l2(1e-5),
                strides=(strd, strd),
            )(shortcut)

            block = Add()([layer, projection_layer])
            if not last:
                block = tf.nn.relu(block)
            return block

        else:
            block = Add()([layer, shortcut])
            if not last:
                block = tf.nn.relu(block)
            return block

    return BLOCK


def ResNet32(in_sh, classes_, activ_last=False, gpu='0'):

    with tf.device("/GPU:"+gpu):

        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input,
            Activation,
            Conv2D,
            BatchNormalization,
        )
        from tensorflow.keras.layers import (
            AveragePooling2D,
            MaxPooling2D,
            Flatten,
            Dense,
        )
        from tensorflow.keras.layers import (
            GlobalAveragePooling2D,
            GlobalMaxPooling2D,
            Dropout,
        )
        from tensorflow.keras.regularizers import l2
        from tensorflow.keras.activations import relu, softmax

        input_layer = Input(shape=in_sh)

        layer = Conv2D(
            16, kernel_size=(3, 3), kernel_regularizer=l2(1e-5), padding="same"
        )(input_layer)

        layer = BatchNormalization()(layer)
        layer = Activation(relu)(layer)

        layer = residual_block(16, 1)(layer)
        layer = residual_block(16, 1)(layer)
        layer = residual_block(16, 1)(layer)
        layer = residual_block(16, 1)(layer)
        layer = residual_block(16, 1)(layer)

        layer = residual_block(32, 2)(layer)
        layer = residual_block(32, 1)(layer)
        layer = residual_block(32, 1)(layer)
        layer = residual_block(32, 1)(layer)
        layer = residual_block(32, 1)(layer)

        layer = residual_block(64, 2)(layer)
        layer = residual_block(64, 1)(layer)
        layer = residual_block(64, 1)(layer)
        layer = residual_block(64, 1)(layer)
        layer = residual_block(64, 1, last=True)(layer)

        layer = GlobalAveragePooling2D()(layer)

        layer = Dense(classes_, kernel_regularizer=l2(1e-5))(layer)
        if activ_last:
            layer = Activation(softmax)(layer)

        model_ = Model(inputs=input_layer, outputs=layer)

        print("\nInput:" + str(input_layer))
        print("Output:" + str(model_.output) + "\n")

        return model_
