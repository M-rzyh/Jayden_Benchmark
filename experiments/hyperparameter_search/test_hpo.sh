#!/bin/bash

python launch_sweep.py \
--algo gpi_ls_discrete \
--env-id MOLunarLanderDefault-v0 \
--ref-point '-101' '-1001' '-101' '-101' \
--sweep-count 100 \
--seed 10 \
--num-seeds 3 \
--config-name gpi-ls \
--train-hyperparams total_timesteps:200000