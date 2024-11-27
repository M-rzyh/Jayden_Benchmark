#!/bin/bash

python3 -u launch_experiment.py \
--algo morld \
--env-id MOHumanoidDR-v5 \
--seed 5 \
--num-timesteps 10000000 \
--gamma 0.99 \
--wandb-group 'domain_randomization' \
--ref-point '-100.0' '-100.0' \
--test-generalization True \
--init-hyperparams shared_buffer:True exchange_every:20000 pop_size:6 "policy_args:{'learning_starts':5000, 'batch_size':256, 'buffer_size':1000000, 'net_arch':[256,256,256,256]}"  \
--train-hyperparams eval_mo_freq:100000  \
--generalization-hyperparams num_eval_weights:100 num_eval_episodes:1 record_video_w_freq:397 \
--test-envs "MOHumanoidDefault-v5,MOHumanoidLight-v5,MOHumanoidHeavy-v5,MOHumanoidLowDamping-v5,MOHumanoidHard-v5" \
# --record-video True \