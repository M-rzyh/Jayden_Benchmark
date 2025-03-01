# MORL-Generalization

MORL-Generalization is a benchmark for evaluating the capabilities of Multi-Objective Reinforcement Learning (MORL) algorithms to generalize across environments.

Our domains are adapted from [MO-Gymnasium](https://github.com/Farama-Foundation/mo-gymnasium) and the implementations of the baseline algorithms are adapted from [MORL-Baselines](https://github.com/LucasAlegre/morl-baselines).

**NOTE:** We will release more instructions on using our software in the coming weeks. Please reach out directly if you need instructions on our code urgently.

## Setup
To install the necessary dependencies, first make sure you have the necessary packages to install [pycddlib](https://pycddlib.readthedocs.io/en/latest/quickstart.html). Then, run the following commands:
```bash
pip install swig
pip install -r requirements.txt
```

## Dataset
The evaluations of 8 state-of-the-art algorithms and SAC on our benchmark domains can be found on [https://wandb.ai/jayden-teoh/MORL-Generalization](https://wandb.ai/jayden-teoh/MORL-Generalization).

## Updates
**[2025/01]** Our paper "On Generalization Across Environments In Multi-Objective Reinforcement Learning" has been accepted at ICLR 2025! 🎉🎉

## Citing

<!-- start citation -->

If you use this repository in your research, please cite:
```bibtex
@inproceedings{
teoh2025on,
title={On Generalization Across Environments In Multi-Objective Reinforcement Learning},
author={Jayden Teoh and Pradeep Varakantham and Peter Vamplew},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=tuEP424UQ5}
}
```
Please also cite [MO-Gymnasium](https://github.com/Farama-Foundation/mo-gymnasium) if you use any of the baseline algorithms for evaluations.
