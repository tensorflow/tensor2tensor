# Tensor2Tensor experimental Model-Based Reinforcement Learning.

The rl package intention is to provide possiblity to run reinforcement
algorithms within Tensorflow's computation graph, in order to do model-based
RL using envoronment models from Tensor2Tensor. It's very experimental
for now and under heavy development.

Currently the only supported algorithm is Proximy Policy Optimization - PPO.

## Sample usage - training in the Pendulum-v0 environment.

```python rl/t2t_rl_trainer.py --problems=Pendulum-v0 --hparams_set continuous_action_base [--output_dir dir_location]```

## Sample usage - training in the PongNoFrameskip-v0 environment.

```python tensor2tensor/rl/t2t_rl_trainer.py --problem stacked_pong --hparams_set atari_base --hparams num_agents=5 [--output_dir dir_location]```

## Sample usage - generation of trajectories data

```python tensor2tensor/bin/t2t-datagen --data_dir=~/t2t_data --tmp_dir=~/t2t_data/tmp --problem=gym_pong_trajectories_from_policy --model_path [model]```
