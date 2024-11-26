#!/bin/bash

python3 -u launch_experiment.py \
--algo pcn \
--env-id MOHopperDR-v5 \
--seed 5 \
--num-timesteps 3000000 \
--gamma 0.99 \
--wandb-group 'domain_randomization' \
--ref-point '-100.0' '-100.0' '-100.0' \
--test-generalization True \
--init-hyperparams "scaling_factor:[0.1, 0.1, 0.1, 0.1]" "max_return:[2700, 3100, 1000]" "net_arch:[256, 256, 256, 256]" \
--train-hyperparams eval_mo_freq:100000 max_buffer_size:1000 num_er_episodes:1000 \
--generalization-hyperparams num_eval_weights:100 num_eval_episodes:1 record_video_w_freq:125 \
--test-envs "MOHopperDefault-v5,MOHopperLight-v5,MOHopperHeavy-v5,MOHopperSlippery-v5,MOHopperLowDamping-v5,MOHopperHard-v5" \
# --record-video True \