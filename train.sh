#!/bin/bash


python3 -u launch_experiment.py \
--algo morld \
--env-id MOHopperDR-v5 \
--num-timesteps 20000000 \
--gamma 0.99 \
--auto-tag False \
--ref-point '0.0' '-20.0' '-20.0' \
--init-hyperparams shared_buffer:True exchange_every:5000 pop_size:10 \
--train-hyperparams test_generalization:True \
--test-envs "MOHopperDefault-v5,MOHopperLight-v5,MOHopperHeavy-v5,MOHopperSlippery-v5,MOHopperLowDamping-v5" \