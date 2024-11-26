#!/bin/bash

python3 -u launch_experiment.py \
--algo sac_continuous \
--env-id MOHalfCheetahDR-v5 \
--seed 92 \
--num-timesteps 5000000 \
--gamma 0.99 \
--wandb-group 'domain_randomization' \
--ref-point '-100.0' '-500.0' \
--test-generalization True \
--init-hyperparams alpha:0.2 batch_size:256 buffer_size:1000000 "net_arch:[256, 256, 256, 256]" \
--train-hyperparams eval_mo_freq:100000 \
--generalization-hyperparams num_eval_weights:100 num_eval_episodes:1 record_video_ep_freq:1001 \
--test-envs "MOHalfCheetahDefault-v5,MOHalfCheetahLight-v5,MOHalfCheetahHeavy-v5,MOHalfCheetahSlippery-v5,MOHalfCheetahHard-v5" \
# --record-video True \