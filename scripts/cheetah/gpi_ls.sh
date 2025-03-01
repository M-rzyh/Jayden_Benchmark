#!/bin/bash

python3 -u launch_experiment.py \
--algo gpi_ls_continuous \
--env-id MOHalfCheetahDR-v5 \
--seed 5 \
--num-timesteps 5000000 \
--gamma 0.99 \
--wandb-group 'domain_randomization' \
--ref-point '-100.0' '-500.0' \
--test-generalization True \
--init-hyperparams per:True gradient_updates:1 batch_size:256 learning_starts:25000 buffer_size:1000000 "net_arch:[256, 256, 256, 256]" \
--train-hyperparams timesteps_per_iter:25000 eval_mo_freq:100000 \
--generalization-hyperparams num_eval_weights:100 num_eval_episodes:1 "generalization_algo:'dr_state_action_history'" history_len:2 \
--test-envs "MOHalfCheetahDefault-v5,MOHalfCheetahLight-v5,MOHalfCheetahHeavy-v5,MOHalfCheetahSlippery-v5,MOHalfCheetahHard-v5" \
# --record-video True \
# --record_video_w_freq:225 \