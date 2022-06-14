### Computationally Efficient Rehearsal for Online Continual Learning ###

This is the TensorFlow2 method implementations as described in the paper:

<br/>
Davalas, Charalampos & Michail, Dimitrios & Diou, Christos & Varlamis, Iraklis & Tserpes, Konstantinos. (2022). Computationally Efficient Rehearsal for Online Continual Learning. 10.1007/978-3-031-06433-3_4.
<br/>



### Background ###
We test our framework for efficient rehearsal, alongside experience replay baselines in order to show that we can reduce computations with little to no effect on overall classification accuracy.

Our Methods are composed by the following:

 * Drift-Activated rehearsal (Where the drift detector activated a rehearsal strategy when the model shows degradation )
 * Dynamic Rehearsal Iterations, Based on misclassification rate
 * Learning Rate Scheduling, Based on misclassification rate
 * Convergence-Based Rehearsal, where the model uses rehearsal repetitions until it converges to the solution.

Our baselines are:

 * Online learning while inducing catastrophic forgetting ` [catf] `
 * Continual Rehearsal learning while repeating N-times rehearsal, per batch ` [conr(N)] `




### Prerequisities ###
An anaconda or virtualenv with python=3.7 is needed in order to run the scripts. Note that for tensorflow-gpu special instructions are provided in the tensorflow page. The anaconda installation can offer the gpu prerequisites (such as cudatoolkit) although the process of installing tensorflow with anaconda is slightly different.

We provide the necessary requirements.txt for the tried-and-true versions of each package.



### Installation/ Instructions ###

Assuming that the environment is a linux terminal or WSL ...

  * git clone project and cd
  * Create and activate conda/virtualenv.
  * install suggested requirements (we provide a requirements.txt) and check if properly installed.
  * python3 run.py [--run-all-methods | --only-drifta | --only-conr | --only-drifta2] (Check the argument list below)
 <details>
    <summary> <b> Argument list (Click here to expand) </b> </summary>

```
    --seed SEED                  seed for reproducibility
    --gpu GPU                    gpu id
    --batch BATCH                stream batch size
    --test-batch TEST_BATCH      test batch size
    --pretrain-epochs PRE_EPOCHS network warm-up epochs

    --lrates LEARNING_RATES [LEARNING_RATES ...]           learning rates
    --er-repeats CONR_N_RH_REPEAT [CONR_N_RH_REPEAT ...]   repeat steps for conr/er
    --pretrain-im-per-class PRETRAIN_NUM_PER_CLASS         number of pretrain data per class
    --buffer-im-per-class BUFFER_NUM_PER_CLASS             number of buffer data per class
    --error-thr ERROR_THR                                  error threshold for hybrid methods
    --max-no-train MAX_NOTRAIN                             idle/no train threshold

    --lr_decay_steps LEARNING_RATE_DECAY_STEPS
    --lr_decay_rate LEARNING_RATE_DECAY_RATE
    --opt OPTIMIZERS [OPTIMIZERS ...]
    --dataset DATASET_NAME
    --random-task-select
    --augment-images

    --drifta                           check the drift activated methods
    --drifta2                          check the two drift detector methods
    --conr                             check the continual rehearsal methods

    --static-lr                        deactivate dynamic rate schedule globally
    --static-rh-repeat                 deactivate dynamic rehearsal repeat globally
    --checkpoints                      store intermediate models
    --stream_batch_n STREAM_BATCH_NUM  stream batch number
    --task_num TASK_NUM                number of tasks

    --stream-task-batch-num TASK_SIZE_IN_BATCHES stream task batch number

    --drifta-max-repeat DRIFTA_MAX_RH_REPEAT [DRIFTA_MAX_RH_REPEAT ...]
                                                      maximum repeat for drift activated methods

    --mix-len MIX_LEN     batch mix ratio new/old
    --lam LAM             lam for ECDD drift detector
    --avg-run-len ARL     average run length for ECDD drift detector
```
</details>


### Acknowlegdements ###

This work is supported by the ``TEACHING'' project that has received funding from the European Union’s Horizon 2020 research and innovation programme under the grant agreement No 871385. The work reflects only the author’s view and the EU Agency is not responsible for any use that may be made of the information it contains

### Citation ###


Thanks in advance for citing us :)
```
@inbook{inbook,
author = {Davalas, Charalampos and Michail, Dimitrios and Diou, Christos and Varlamis, Iraklis and Tserpes, Konstantinos},
year = {2022},
month = {05},
pages = {39-49},
title = {Computationally Efficient Rehearsal for Online Continual Learning},
isbn = {978-3-031-06432-6},
doi = {10.1007/978-3-031-06433-3_4}
}

```
