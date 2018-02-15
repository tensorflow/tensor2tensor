# Tensor2Tensor experimental Model-Based Reinforcement Learning.

The rl package intention is to provide possiblity to run reinforcement
algorithms within Tensorflow's computation graph, in order to do model-based
RL using envoronment models from Tensor2Tensor. It's very experimental
for now and under heavy development.

Currently the only supported algorithm is Proximy Policy Optimization - PPO.

## Sample usage - training in Pendulum-v0 environment.

```python rl/t2t_rl_trainer.py --problems=Pendulum-v0 --hparams_set continuous_action_base [--output_dir dir_location]```
