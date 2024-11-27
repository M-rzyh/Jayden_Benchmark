#!/bin/bash

python3 -u launch_experiment.py \
--algo pcn \
--env-id MOSuperMarioBrosZeroShot-v2 \
--seed 92 \
--num-timesteps 3000000 \
--gamma 0.99 \
--ref-point '-100.0' '-100.0' '-100.0' \
--wandb-group 'domain_randomization' \
--wandb-tags 'mario zero shot' \
--test-generalization True \
--init-hyperparams "scaling_factor:[0.1, 0.1, 0.1, 0.1]" "max_return:[100, 100, 200]" "net_arch:[512, 512]" "batch_size:64" \
--train-hyperparams eval_mo_freq:100000 max_buffer_size:100000 num_er_episodes:200 \
--generalization-hyperparams num_eval_weights:32 num_eval_episodes:1 \
--test-envs "MOSuperMarioBros-1-2-v2,MOSuperMarioBros-3-2-v2,MOSuperMarioBros-3-3-v2,MOSuperMarioBros-4-3-v2,MOSuperMarioBros-5-2-v2,MOSuperMarioBros-5-3-v2,MOSuperMarioBros-7-3-v2,MOSuperMarioBros-8-1-v2" \
# --record-video True \