# Tensor2Tensor experimental Model-Based Reinforcement Learning.

**Note**: Experimental and under development.

The `rl` package provides the ability to run model-based reinforcement learning
algorithms using models trained with Tensor2Tensor.

Currently this entails alternating model training and agent training using
Proximal Policy Optimization (PPO). See `trainer_model_based.py`.

As a baseline, you can also run PPO without the model using
`trainer_model_free.py`.

## Model-based training

Alternate training a world model and a PPO agent within that model using the
base hyperparameters on Freeway:

```
python -m tensor2tensor.rl.trainer_model_based \
  --output_dir=$OUT_DIR \
  --loop_hparams_set=rl_modelrl_base \
  --loop_hparams='game=freeway'
```

All hyperparameter sets are defined in `trainer_model_based.py` and are derived
from `rl_modelrl_base`.

The hyperparameters for the environment model and agent are nested within the
`loop_hparams` by name. For example:

```
  generative_model="next_frame_basic",
  generative_model_params="next_frame_pixel_noise",
  ppo_params="ppo_pong_base",
```

## Model-free training

**TODO(piotrmilos): Update**

Training an agent in `Pendulum-v0`:

```
python -m tensor2tensor.rl.trainer_model_free \
  --problem=Pendulum-v0 \
  --hparams_set ppo_continuous_action_base \
  --output_dir $OUT_DIR
```

Training an agent in `PongNoFrameskip-v0`:

```
python -m tensor2tensor.rl.trainer_model_free \
  --problem stacked_pong \
  --hparams_set ppo_atari_base \
  --hparams num_agents=5 \
  --output_dir dir_location
```

## Model training on random trajectories

Generate trajectories with a random policy:

```
python -m tensor2tensor.rl.datagen_with_agent \
  --data_dir=$HOME/t2t/data \
  --tmp_dir=$HOME/t2t/tmp \
  --game=pong \
  --num_env_steps=30000
```

Train model on trajectories:

```
python -m tensor2tensor.bin.t2t_trainer \
  --data_dir=$HOME/t2t/data \
  --output_dir=$HOME/t2t/train/pong_model \
  --problem=gym_pong_random \
  --model=next_frame_basic \
  --hparams_set=next_frame
```


## Collect trajectories using a trained agent

```
python -m tensor2tensor.rl.datagen_with_agent \
  --data_dir=$HOME/t2t/data \
  --tmp_dir=$HOME/t2t/tmp \
  --game=pong \
  --num_env_steps=30000 \
  --agent_policy_path=$AGENT_CKPT_PATH
```

Add `--eval` if you want to evaluate the agent against the environment instead
of generating trajectories for training the world model.
