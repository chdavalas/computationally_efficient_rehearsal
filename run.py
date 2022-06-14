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
from methods import OnlineTraining, ContinuousRehearsal, ContinuousRehearsalConverge
from methods import DriftActivatedRehearsal, DriftActivatedRehearsalConverge


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

def stream_testing(model, dts_list, verbose=0):
    acc_list = []
    for dts in dts_list:
        _, acc = model.evaluate(dts, verbose=verbose)
        acc_list += [round(acc, 3)]
    return acc_list

def create_tasks(
    stream_size,
    task_size,
    task_classes,
    random_task_selection=False,
    random_task_sizes=False,
    random_task_sizes_sigma=3,
    dataset_name="cifar10",
    train_batch_size=32,
    test_batch_size=256,
    seed=32221100,
    shuffle_buffer_size=60000,
):
    avail_tasks = {}
    test_stream = []
    for i, task in enumerate(task_classes):
        train_task, test_task, _, _ = custom_dataset(
            dataset_name, task, task, -1, -1, seed, True
        )
        avail_tasks[i] = train_task
        test_task = test_task.cache().batch(test_batch_size)
        test_stream.append(test_task)

    number_of_tasks = len(avail_tasks)
    tasks_sizes = []
    tasks = []

    remaining_size = stream_size

    i = 0
    while remaining_size > 0:
        if random_task_selection:
            current_task = random.randrange(number_of_tasks)
        else:
            current_task = i % number_of_tasks

        if remaining_size < task_size:
            current_task_size = remaining_size
        else:
            if random_task_sizes:
                current_task_size = int(
                    random.gauss(task_size, sigma=random_task_sizes_sigma)
                )
            else:
                current_task_size = task_size

        if current_task_size <= 0:
            continue

        chosen_task = avail_tasks[current_task]
        tasks_sizes += [current_task_size]
        tasks += [current_task]

        chosen_task_stream = chosen_task.shuffle(shuffle_buffer_size, seed=seed).take(
            current_task_size
        )
        if stream_size == remaining_size:
            train_stream = chosen_task_stream
        else:
            train_stream = train_stream.concatenate(chosen_task_stream)

        remaining_size -= current_task_size
        i += 1

    actual_task_per_step = []
    for cur_size, cur_task in zip(tasks_sizes, tasks):
        cur_steps = cur_size // train_batch_size
        actual_task_per_step.extend([cur_task] * cur_steps)
        total_number_of_steps = len(actual_task_per_step)

    return (
        number_of_tasks,
        tasks,
        tasks_sizes,
        total_number_of_steps,
        actual_task_per_step,
        train_stream,
        test_stream,
    )


def clone_initial_model(initial_model, optimizer, loss):
    model = tf.keras.models.clone_model(initial_model)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    model.set_weights(initial_model.get_weights())
    return model


def optimizer_factory(
    optimizer,
    initial_learning_rate,
    decay_rate,
    decay_steps,
):
    learning_rate_fn = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
    )

    if optimizer == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate_fn)
    elif optimizer == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    raise ValueError("Not supported")


def main():

    parser = argparse.ArgumentParser(description='Computationally Efficient Rehearsal')

    parser.add_argument('--seed', dest='SEED', type=int, default=32221100, help='seed for reproducibility') 

    parser.add_argument('--gpu',  dest='GPU', type=str, default="0", help='gpu id') 

    parser.add_argument('--batch', dest='BATCH', type=int, default=32, help='stream batch size') 

    parser.add_argument('--test-batch', dest='TEST_BATCH', type=int, default=256, help='test batch size') 

    parser.add_argument('--pretrain-epochs', dest='PRE_EPOCHS', type=int, default=100, help='network warm-up epochs') 

    parser.add_argument('--lrates',     dest='LEARNING_RATES', type=float, default=[0.01], nargs='+', help='learning rates') 

    parser.add_argument('--er-repeats', dest='CONR_N_RH_REPEAT', type=int,   default=[1,10,25,50], nargs='+', help='repeat steps for conr/er') 

    parser.add_argument('--pretrain-im-per-class', dest='PRETRAIN_NUM_PER_CLASS', type=int, default=1500, help='number of pretrain data per class') 

    parser.add_argument('--buffer-im-per-class',   dest='BUFFER_NUM_PER_CLASS', type=int, default=500, help='number of buffer data per class') 

    parser.add_argument('--error-thr', dest='ERROR_THR', type=float, default=0.2, help='error threshold for hybrid methods') 

    parser.add_argument('--max-no-train', dest='MAX_NOTRAIN', type=int, default=20, help='idle/no train threshold') 

    parser.add_argument('--lr_decay_steps', dest='LEARNING_RATE_DECAY_STEPS', type=float, default=1.0) 

    parser.add_argument('--lr_decay_rate', dest='LEARNING_RATE_DECAY_RATE', type=float, default=0.05) 

    parser.add_argument('--opt', dest='OPTIMIZERS', type=str, default=["sgd"], nargs='+') 

    parser.add_argument('--dataset', dest='DATASET_NAME', type=str, default="cifar10") 

    parser.add_argument('--random-task-select', dest='USE_RANDOM_TASK_SELECTION', action="store_true") 

    parser.add_argument('--random-task-size', dest='USE_RANDOM_TASK_SIZE', action="store_true") 

    parser.add_argument('--augment-images', dest='AUG', action="store_true") 

    parser.add_argument('--drifta', dest='DRIFTA', action="store_true", help="check the drift activated methods") 

    parser.add_argument('--drifta2', dest='DRIFTA2', action="store_true", help="check the two drift detector methods") 

    parser.add_argument('--conr', dest='CONR', action="store_true", help="check the continual rehearsal methods") 

    parser.add_argument('--all-methods', dest='ALL', action="store_true", help="check all methods") 

    parser.add_argument('--static-lr', dest='DYN_LR', action="store_false", help="deactivate dynamic rate schedule globally") 
    
    parser.add_argument('--static-rh-repeat', dest='DYN_RH_REP', action="store_false", help="deactivate dynamic rehearsal repeat globally") 

    parser.add_argument('--checkpoints',      dest='SAVE_INTERMEDIATE_MODELS', action="store_true", help='store intermediate models') 

    parser.add_argument('--stream_batch_n', dest='STREAM_BATCH_NUM', type=int, default=1500, help='stream batch number') 

    parser.add_argument('--task_num', dest='TASK_NUM', type=int, default=5, help='number of tasks') 

    parser.add_argument('--stream-task-batch-num', dest='TASK_SIZE_IN_BATCHES', type=int, default=100, help='stream task batch number') 

    parser.add_argument('--drifta-max-repeat', dest='DRIFTA_MAX_RH_REPEAT', default=[50], nargs='+', help='maximum repeat for drift activated methods') 

    parser.add_argument('--mix-len', dest='MIX_LEN', type=int, default=16, help='batch mix ratio new/old') 

    parser.add_argument('--lam', dest='LAM', type=float, default=0.2, help='lam for ECDD drift detector') 

    parser.add_argument('--avg-run-len', dest='ARL', type=int, default=100, help='average run length for ECDD drift detector') 

    args = parser.parse_args()

    if not args.CONR and not args.DRIFTA and not args.DRIFTA2 and not args.ALL:
        print("Use one of the following flags --conr, --drifta, --drifta2 OR --all")
        quit()

    SEED                      = args.SEED
    BATCH                     = args.BATCH
    PRE_EPOCHS                = args.PRE_EPOCHS
    LEARNING_RATES            = args.LEARNING_RATES
    LEARNING_RATE_DECAY_STEPS = args.LEARNING_RATE_DECAY_STEPS
    LEARNING_RATE_DECAY_RATE  = args.LEARNING_RATE_DECAY_RATE
    OPTIMIZERS                = args.OPTIMIZERS
    DYN_LR                    = args.DYN_LR
    DYN_RH_REP                = args.DYN_RH_REP
    
    PRETRAIN_NUM_PER_CLASS = args.PRETRAIN_NUM_PER_CLASS
    BUFFER_NUM_PER_CLASS   = args.BUFFER_NUM_PER_CLASS

    TEST_BATCH               = args.TEST_BATCH
    CONR_N_RH_REPEAT         = args.CONR_N_RH_REPEAT 
    DRIFTA_MAX_RH_REPEAT     = args.DRIFTA_MAX_RH_REPEAT # drift activated maximum repeat
    MIX_LEN                  = args.MIX_LEN
    SAVE_INTERMEDIATE_MODELS = args.SAVE_INTERMEDIATE_MODELS 

    # Drift parameters
    LAM = args.LAM  # Lambda hyper parameter. Optimal is 0.2 according to the respective paper

    # According to the respective paper 100,400,1000 are the basic variables for this one
    # ARL (Average run length) , configures the sensitivity of drift detection (higher means less sensitive)

    ARL = args.ARL

    # Hybrid method parameters
    ERROR_THR =  args.ERROR_THR 
    # If Z_t error exceeds this value, then train no matter what

    MAX_NOTRAIN = args.MAX_NOTRAIN
    # If, after MAX_NOTRAIN mini-batches the drift has not been activated, then train no matter what again.
    DATASET_NAME = args.DATASET_NAME

    MODEL_CACHE = "model_cache/"
    NAME_OF_INIT_MODEL = "model_cache/pretrained-{}{}".format(DATASET_NAME, SEED)

    BATCHES_IN_DATASET = 50000 // BATCH 
    SIZE_OF_STREAM = BATCH * args.STREAM_BATCH_NUM
    SIZE_OF_TASK   = BATCH * args.TASK_SIZE_IN_BATCHES
    USE_RANDOM_TASK_SELECTION = args.USE_RANDOM_TASK_SELECTION
    USE_RANDOM_TASK_SIZE      = args.USE_RANDOM_TASK_SIZE 
    RANDOM_TASK_SIZE_SIGMA    = BATCH * 5

    if DATASET_NAME=="cifar10" or DATASET_NAME=="mnist":
        CLASSES=10
    else:
        quit("Unknown dataset name")

    ALL_CLASSES  = [i for i in range(CLASSES)]
    TASK_CLASSES = [list(range(i,int(CLASSES/args.TASK_NUM)+i)) for i in range(0, CLASSES, int(CLASSES/args.TASK_NUM))]

    print("Starting")

    LOSS = tf.keras.losses.CategoricalCrossentropy()

    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if len(physical_devices) != 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Number of available GPUs: ", len(physical_devices))

    #print_all_parameters()

    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)


    print("Creating rehearsal buffer")
    buffer_dts, _, n_classes, img_sh = custom_dataset(
        DATASET_NAME, ALL_CLASSES, [], BUFFER_NUM_PER_CLASS, -1, SEED, True
    )
    buffer_size = BUFFER_NUM_PER_CLASS * len(n_classes)

    for j, (im, la) in enumerate(buffer_dts.batch(buffer_size)):
        if j == 0:
            buffer_im = im
            buffer_la = la
        else:
            buffer_im = tf.concat([buffer_im, im], axis=0)
            buffer_la = tf.concat([buffer_la, la], axis=0)


    print("Creating image sequence")
    (
        number_of_tasks,
        tasks,
        tasks_sizes,
        total_number_of_steps,
        actual_task_per_step,
        train_stream,
        test_stream,
    ) = create_tasks(
        SIZE_OF_STREAM,
        SIZE_OF_TASK,
        TASK_CLASSES,
        random_task_selection=USE_RANDOM_TASK_SELECTION,
        random_task_sizes=USE_RANDOM_TASK_SIZE,
        random_task_sizes_sigma=RANDOM_TASK_SIZE_SIGMA,
    )


    print("Total tasks: {}".format(number_of_tasks))
    print("Sequence of tasks: {}".format(tasks))
    print("Each task size in images: {}".format(tasks_sizes))
    print("Total number of steps: {}".format(total_number_of_steps))


    print("INIT & PRETRAIN MODELS")

    if not tf.io.gfile.exists(NAME_OF_INIT_MODEL):
        print("A new model has been created, offline training will commence...")
        print("Creating pretraining dataset")
        pretrain_dts, _, pretrain_n_classes, pretrain_img_shape = custom_dataset(
            DATASET_NAME, ALL_CLASSES, [], PRETRAIN_NUM_PER_CLASS, -1, SEED, True
        )
        init_model = ResNet32(pretrain_img_shape, len(pretrain_n_classes), args.GPU)
        init_model.compile(loss=LOSS, metrics=["accuracy"])
        init_model_opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

        with tf.device("/GPU:"+args.GPU):
            dataset_training(
                init_model,
                pretrain_dts.batch(BATCH),
                PRE_EPOCHS,
                opt=init_model_opt,
                loss=LOSS,
                seed_=SEED,
                shuf=True,
            )
            init_model.save(NAME_OF_INIT_MODEL, overwrite=True)
    else:
        print("Loading model from: {}".format(NAME_OF_INIT_MODEL))
        init_model = tf.keras.models.load_model(NAME_OF_INIT_MODEL)

    print("Initializing score board")
    scoreboard = defaultdict(list)
    scoreboard["task"] = actual_task_per_step


    # Create methods
    print("Creating separate model per method")
    METHODS = []

    #
    # Online Training With Catastrophic Forgetting
    #
    for learning_rate in LEARNING_RATES:
        for optimizer_name in OPTIMIZERS:
            optimizer = optimizer_factory(
                optimizer_name, learning_rate, decay_steps=LEARNING_RATE_DECAY_STEPS, decay_rate=LEARNING_RATE_DECAY_RATE
            )
            model = clone_initial_model(init_model, optimizer, LOSS)
            METHODS += [
                OnlineTraining(
                    name="catf_{}_{}".format(optimizer_name, learning_rate),
                    model=model,
                    optimizer=optimizer,
                    loss=LOSS,
                    scoreboard=scoreboard,
                    repeat=1,
                    seed=SEED,
                    augment_images=args.AUG,
                )
            ]

    #
    # continuous rehearsal
    #
    if args.CONR or args.ALL:
        for learning_rate in LEARNING_RATES:
            for optimizer_name in OPTIMIZERS:
                for conr_repeat in CONR_N_RH_REPEAT:
                    optimizer = optimizer_factory(
                        optimizer_name, learning_rate, decay_steps=LEARNING_RATE_DECAY_STEPS, decay_rate=LEARNING_RATE_DECAY_RATE
                    )
                    model = clone_initial_model(init_model, optimizer, LOSS)
                    name = "conr{}_{}_{}".format(conr_repeat, optimizer_name, learning_rate)
                    METHODS += [
                        ContinuousRehearsal(
                            name=name,
                            model=model,
                            optimizer=optimizer,
                            loss=LOSS,
                            rehearsal_buffer_images=buffer_im,
                            rehearsal_buffer_labels=buffer_la,
                            rehearsal_repeats=conr_repeat,
                            train_every_steps=1,
                            scoreboard=scoreboard,
                            seed=SEED,
                            mix_len=MIX_LEN,
                            augment_images=args.AUG,
                        )
                    ]

                # Continuous rehearsal until convergence
                METHODS += [
                    ContinuousRehearsalConverge(
                        name="cont_conv_{}_{}".format(optimizer_name, learning_rate),
                        model=model,
                        optimizer=optimizer,
                        loss=LOSS,
                        rehearsal_buffer_images=buffer_im,
                        rehearsal_buffer_labels=buffer_la,
                        train_every_steps=1,
                        scoreboard=scoreboard,
                        seed=SEED,
                        mix_len=MIX_LEN,
                        augment_images=args.AUG,
                        alpha_short=0.5,
                        alpha_long=0.05,
                        eps=0.005,
                    )
                ]


    if args.DRIFTA or args.ALL:

        # drift activated
        for learning_rate in LEARNING_RATES:
            for optimizer_name in OPTIMIZERS:
                for drift_repeat in DRIFTA_MAX_RH_REPEAT:
                    # plain drift activated
                    optimizer = optimizer_factory(
                        optimizer_name, learning_rate, decay_steps=LEARNING_RATE_DECAY_STEPS, decay_rate=LEARNING_RATE_DECAY_RATE
                    )
                    model = clone_initial_model(init_model, optimizer, LOSS)
                    METHODS += [
                        DriftActivatedRehearsal(
                            name="drifta_{}_{}_{}".format(drift_repeat, optimizer_name, learning_rate),
                            model=model,
                            optimizer=optimizer,
                            loss=LOSS,
                            rehearsal_buffer_images=buffer_im,
                            rehearsal_buffer_labels=buffer_la,
                            rehearsal_repeats=drift_repeat,
                            scoreboard=scoreboard,
                            seed=SEED,
                            mix_len=MIX_LEN,
                            err_thr=ERROR_THR,
                            max_notrain=MAX_NOTRAIN,
                            use_rehearsal_drift_detector=False,
                            augment_images=args.AUG,
                            dynamic_initial_learning_rate=DYN_LR,
                            dynamic_rehearsal_repeats=DYN_RH_REP,
                            avg_run_len=ARL,
                        )
                    ]

                # Drift activated rehearsal until convergence
                optimizer = optimizer_factory(
                    optimizer_name, learning_rate, decay_steps=LEARNING_RATE_DECAY_STEPS, decay_rate=LEARNING_RATE_DECAY_RATE
                )
                model = clone_initial_model(init_model, optimizer, LOSS)
                METHODS += [
                    DriftActivatedRehearsalConverge(
                        name="drifta_conv_{}_{}".format(optimizer_name, learning_rate),
                        model=model,
                        optimizer=optimizer,
                        loss=LOSS,
                        rehearsal_buffer_images=buffer_im,
                        rehearsal_buffer_labels=buffer_la,
                        scoreboard=scoreboard,
                        seed=SEED,
                        mix_len=MIX_LEN,
                        err_thr=ERROR_THR,
                        max_notrain=MAX_NOTRAIN,
                        use_rehearsal_drift_detector=False,
                        dynamic_initial_learning_rate=DYN_LR,
                        rehearsal_drift_detector_batch=10,
                        augment_images=args.AUG,
                        alpha_short=0.5,
                        alpha_long=0.05,
                        eps=0.005,
                        avg_run_len=ARL,
                    )
                ]


    if args.DRIFTA2 or args.ALL:

        # two drift detectors
        for learning_rate in LEARNING_RATES:
            for optimizer_name in OPTIMIZERS:

                # 2drifta
                optimizer = optimizer_factory(
                    optimizer_name, learning_rate, decay_steps=LEARNING_RATE_DECAY_STEPS, decay_rate=LEARNING_RATE_DECAY_RATE
                )
                model = clone_initial_model(init_model, optimizer, LOSS)
                METHODS += [
                    DriftActivatedRehearsal(
                        name="2drifta_{}_{}".format(optimizer_name, learning_rate),
                        model=model,
                        optimizer=optimizer,
                        loss=LOSS,
                        rehearsal_buffer_images=buffer_im,
                        rehearsal_buffer_labels=buffer_la,
                        rehearsal_repeats=DRIFTA_MAX_RH_REPEAT,
                        scoreboard=scoreboard,
                        seed=SEED,
                        mix_len=MIX_LEN,
                        err_thr=ERROR_THR,
                        max_notrain=MAX_NOTRAIN,
                        use_rehearsal_drift_detector=True,
                        augment_images=args.AUG,
                        dynamic_initial_learning_rate=DYN_LR,
                        dynamic_rehearsal_repeats=DYN_RH_REP,
                        avg_run_len=ARL,
                    )
                ]


                # 2drifta converge
                optimizer = optimizer_factory(
                    optimizer_name, learning_rate, decay_steps=LEARNING_RATE_DECAY_STEPS, decay_rate=LEARNING_RATE_DECAY_RATE
                )
                model = clone_initial_model(init_model, optimizer, LOSS)
                METHODS += [
                    DriftActivatedRehearsalConverge(
                        name="2drifta_conv_{}_{}".format(optimizer_name, learning_rate),
                        model=model,
                        optimizer=optimizer,
                        loss=LOSS,
                        rehearsal_buffer_images=buffer_im,
                        rehearsal_buffer_labels=buffer_la,
                        scoreboard=scoreboard,
                        seed=SEED,
                        mix_len=MIX_LEN,
                        err_thr=ERROR_THR,
                        max_notrain=MAX_NOTRAIN,
                        use_rehearsal_drift_detector=True,
                        dynamic_initial_learning_rate=DYN_LR,
                        rehearsal_drift_detector_batch=10,
                        augment_images=args.AUG,
                        alpha_short=0.5,
                        alpha_long=0.05,
                        eps=0.005,
                        avg_run_len=ARL,
                    )
                ]




    print("Computing pretrained model accuracies on all tasks")
    pretrained_acc = stream_testing(init_model, test_stream)
    print("Pretrained model accuracy on all tasks: {}".format(pretrained_acc))

    for t in range(number_of_tasks):
        scoreboard["acc_pretrain_T{}".format(t)] = [
            pretrained_acc[t]
        ] * total_number_of_steps


    print("TRAIN & TEST")

    # Save model every 100 batches
    save_step = 100
    with tf.device("/GPU:"+args.GPU):

        for cur_batch, (images, labels) in enumerate(train_stream.batch(BATCH)):
            # run all methods
            for method in METHODS:
                method.update(images, labels)

            # test all methods
            for method in METHODS:
                accuracy_per_task = stream_testing(method.model, test_stream)
                print("\n####################:")
                print(
                    "Test method={}, batch={}, acc={}, avg_acc={}".format(
                        method.name,
                        cur_batch,
                        accuracy_per_task,
                        round(mean(accuracy_per_task), 3),
                    )
                )
                sys.stdout.flush()
                for t in range(number_of_tasks):
                    scoreboard["acc_{}_T{}".format(method.name, t)].append(
                        accuracy_per_task[t]
                    )
                if ((cur_batch + 1) % save_step == 0) and (SAVE_INTERMEDIATE_MODELS):
                    method.model.save(
                        MODEL_CACHE + method.name + "_batch_{}".format(cur_batch)
                    )


    # convert from dict to dataframe
    df = pd.DataFrame.from_dict(scoreboard)

    # compute average accuracy
    for method in METHODS:
        df["acc_avg_{}".format(method.name)] = df[
            ["acc_{}_T{}".format(method.name, t) for t in range(number_of_tasks)]
        ].mean(axis=1)

    df["acc_avg_pretrain"] = df[
        ["acc_pretrain_T{}".format(t) for t in range(number_of_tasks)]
    ].mean(axis=1)

    # compute forgetting
    for method in METHODS:
        for t in range(number_of_tasks):
            max_acc = pretrained_acc[t]
            forgetting = []

            for step in range(total_number_of_steps):
                value = df.at[step, "acc_{}_T{}".format(method.name, t)]
                diff = 0 if value >= max_acc else max_acc - value
                forgetting.append(diff)
                if value > max_acc:
                    max_acc = value

            df["forget_{}_T{}".format(method.name, t)] = forgetting

        df["forget_avg_{}".format(method.name)] = df[
            ["forget_{}_T{}".format(method.name, t) for t in range(number_of_tasks)]
        ].mean(axis=1)


    results_filename = "{}.csv".format(strftime("d%d_m%m_y%Y_%H%M%S"))
    print("Outputing results to file: {}".format(results_filename))
    df.to_csv(results_filename, index=True, index_label="step")

    args_dict = vars(args)
    print(args_dict)
    with open("args_"+results_filename+".txt", 'w') as f:
        for key, value in args_dict.items():
            f.write('%s:%s\n' % (key,value))

if __name__ == "__main__":
    main()

