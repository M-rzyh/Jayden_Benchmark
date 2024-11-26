#!/bin/bash

python3 -u launch_experiment.py \
--algo morld \
--env-id MOHalfCheetahDR-v5 \
--seed 5 \
--num-timesteps 5000000 \
--gamma 0.99 \
--wandb-group 'domain_randomization' \
--wandb-tags 'new_cheetah' \
--ref-point '-100.0' '-500.0' \
--test-generalization True \
--init-hyperparams shared_buffer:True exchange_every:10000 pop_size:6 "weight_adaptation_method:'PSA'" "policy_args:{'learning_starts':25000, 'batch_size':256, 'buffer_size':1000000, 'net_arch':[256,256,256,256]}" \
--train-hyperparams eval_mo_freq:100000  \
--generalization-hyperparams num_eval_weights:100 num_eval_episodes:1 "algo_suffix:'_PSA'" \
--test-envs "MOHalfCheetahDefault-v5,MOHalfCheetahLight-v5,MOHalfCheetahHeavy-v5,MOHalfCheetahSlippery-v5,MOHalfCheetahHard-v5" \
# --record-video True \