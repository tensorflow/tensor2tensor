# Tensor2Tensor experimental Reinforcement Learning.

The rl package intention is to provide possiblity to run reinforcement
algorithms within Tensorflow's computation graph. It's very experimental
for now and under heavy development.

Currently the only supported algorithm is Proximy Policy Optimization - PPO.

## Sample usage - training in Pendulum-v0 environment.

```python rl/t2t_rl_trainer.py --hparams_set pendulum [--output_dir dir_location]```
