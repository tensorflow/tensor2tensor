# Tensor2Tensor experimental Model-Based Reinforcement Learning.

The rl package intention is to provide possiblity to run reinforcement
algorithms within Tensorflow's computation graph, in order to do model-based
RL using envoronment models from Tensor2Tensor. It's very experimental
for now and under heavy development.

Currently the only supported algorithm is Proximy Policy Optimization - PPO.

# Sample usages

## Training agent in the Pendulum-v0 environment.

```
python rl/t2t_rl_trainer.py \
  --problems=Pendulum-v0 \
  --hparams_set continuous_action_base \
  [--output_dir dir_location]
```

## Training agent in the PongNoFrameskip-v0 environment.

```
python tensor2tensor/rl/t2t_rl_trainer.py \
  --problem stacked_pong \
  --hparams_set atari_base \
  --hparams num_agents=5 \
  [--output_dir dir_location]
```

## Generation of trajectories data

```
python tensor2tensor/bin/t2t-datagen \
  --data_dir=~/t2t_data \
  --tmp_dir=~/t2t_data/tmp \
  --problem=gym_pong_trajectories_from_policy \
  --model_path [model]
```

## Training model for frames generation based on randomly played games

```
python tensor2tensor/bin/t2t-trainer \
  --generate_data \
  --data_dir=~/t2t_data \
  --output_dir=~/t2t_data/output \
  --problems=gym_pong_random5k \
  --model=basic_conv_gen \
  --hparams_set=basic_conv_small \
  --train_steps=1000 \
  --eval_steps=10
```
