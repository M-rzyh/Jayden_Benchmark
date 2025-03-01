#!/bin/bash

python3 -u launch_experiment.py \
--algo morld \
--env-id MOHopperDR-v5 \
--seed 5 \
--num-timesteps 3000000 \
--gamma 0.99 \
--wandb-group 'domain_randomization' \
--ref-point '-100.0' '-100.0' '-100.0' \
--test-generalization True \
--init-hyperparams shared_buffer:True exchange_every:20000 pop_size:6 "weight_adaptation_method:'PSA'" "policy_args:{'alpha':0.2, 'learning_starts':10000, 'batch_size':256, 'buffer_size':1000000, 'net_arch':[256,256,256,256]}"  \
--train-hyperparams eval_mo_freq:100000  \
--generalization-hyperparams num_eval_weights:100 num_eval_episodes:1 "algo_suffix:'_PSA'" "generalization_algo:'dr_state_action_history'" history_len:2 \
--test-envs "MOHopperDefault-v5,MOHopperLight-v5,MOHopperHeavy-v5,MOHopperSlippery-v5,MOHopperLowDamping-v5,MOHopperHard-v5" \
# --record-video True \
# --record_video_w_freq:203 \