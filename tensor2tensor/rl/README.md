# Tensor2Tensor Model-Based Reinforcement Learning.

The `rl` package allows to run reinforcement learning algorithms,
both model-free (e.g., [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347), train with `trainer_model_free.py`) and model-based ones ([SimPLe](https://arxiv.org/abs/1903.00374), train with `trainer_model_based.py`).

You should be able to reproduce the [Model-Based Reinforcement Learning for Atari](https://arxiv.org/abs/1903.00374) results. [These videos](https://sites.google.com/corp/view/modelbasedrlatari/home) show what to expect from the final models.

To use this package, we recommend Tensorflow 1.13.1 and T2T version 1.13.1.
You also need to install the Atari dependencies for OpenAI Gym:

```
pip install gym[atari]
```

[This iPython notebook](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t-rl.ipynb) provides a quick start if you want to check out the videos.


## Play using a pre-trained policy

We provide a set of pretrained policies and models you can use. To evaluate and
generate videos for a pretrained policy on Pong:

```
OUTPUT_DIR=~/t2t_train/pong_pretrained
python -m tensor2tensor.rl.evaluator \
  --loop_hparams_set=rlmb_long_stochastic_discrete \
  --loop_hparams=game=pong \
  --policy_dir=gs://tensor2tensor-checkpoints/modelrl_experiments/train_sd/142/policy \
  --eval_metrics_dir=$OUTPUT_DIR \
  --debug_video_path=$OUTPUT_DIR \
  --num_debug_videos=4
```

By default, it will run a grid of different evaluation settings (sampling
temperatures and whether to do initial rollouts). You can override those
settings:

```
  --loop_hparams=game=pong,eval_max_num_noops=0,eval_sampling_temps=[0.0]
```

TensorBoard metrics are exported to the `eval_metrics_dir`. To view them, run:

```
tensorboard --logdir=~/t2t_train/pong_pretrained
```

Description of player controls and flags can be found in `tensor2tensor/rl/player.py`.


## Train your policy (model-free training)

Training model-free on Pong:

```
python -m tensor2tensor.rl.trainer_model_free \
  --hparams_set=rlmf_base \
  --hparams=game=pong \
  --output_dir=~/t2t_train/mf_pong
```

Hyperparameter sets are defined in `tensor2tensor/models/research/rl.py`. You
can override them using the `hparams` flag, e.g.

```
  --hparams=game=kung_fu_master,frame_stack_size=5
```

As in model-based training, the periodic evaluation runs with timestep limit
of 1000. To do full evaluation after training, run:

```
OUTPUT_DIR=~/t2t_train/mf_pong
python -m tensor2tensor.rl.evaluator \
  --loop_hparams_set=rlmf_base \
  --hparams=game=pong \
  --policy_dir=$OUTPUT_DIR \
  --eval_metrics_dir=$OUTPUT_DIR/full_eval_metrics
```

## World Model training (with random trajectories)

The simplest way to train your own world model is to use random trajectories.
Then you can train a policy on it as described next.

To train a deterministic model:

```
python -m tensor2tensor.rl.trainer_model_based \
  --loop_hparams_set=rlmb_base \
  --loop_hparams=game=pong,epochs=1,ppo_epochs_num=0 \
  --output_dir=~/t2t_train/mb_det_pong_random
```

To train a stochastic discrete model (it will require more time and memory):

```
python -m tensor2tensor.rl.trainer_model_based \
  --loop_hparams_set=rlmb_base_stochastic_discrete \
  --loop_hparams=game=pong,epochs=1,ppo_epochs_num=0 \
  --output_dir=~/t2t_train/mb_sd_pong_random
```

## Playing in the world model

To assess world model quality you can play in it, as in an Atari emulator
(you need a machine with GPU for this). First install `pygame`:

```
pip install pygame
```

Then you can run the player, specifying a path to world model checkpoints:

```
OUTPUT_DIR=~/t2t_train/mb_sd_pong_pretrained
mkdir -p $OUTPUT_DIR
gsutil -m cp -r \
  gs://tensor2tensor-checkpoints/modelrl_experiments/train_sd/142/world_model \
  $OUTPUT_DIR/
python -m tensor2tensor.rl.player \
  --wm_dir=$OUTPUT_DIR/world_model \
  --loop_hparams_set=rlmb_base_stochastic_discrete \
  --loop_hparams=game=pong \
  --game_from_filenames=False \
  --zoom=3 \
  --fps=5
```

The screen is split into 3 columns: frame from the world model, corresponding
frame from the real environment and the difference between the two. Use WSAD
and space to control the agent. The model will likely diverge quickly, press X
to reset it using the current state of the real environment. Note that frames
fed to the model were likely never seen by it during training, so the model's
performance will be worse than during the policy training.

For more details on controls and flags see `tensor2tensor/rl/player.py`.


## Model-based training with pre-trained world models

To train a policy with a pretrained world model (requires Google Cloud SDK):

```
OUTPUT_DIR=~/t2t_train/mb_sd_pong_pretrained
mkdir -p $OUTPUT_DIR
gsutil -m cp -r \
  gs://tensor2tensor-checkpoints/modelrl_experiments/train_sd/142/world_model \
  $OUTPUT_DIR/
python -m tensor2tensor.rl.trainer_model_based \
  --loop_hparams_set=rlmb_base_stochastic_discrete \
  --loop_hparams=game=pong,epochs=1,model_train_steps=0 \
  --eval_world_model=False \
  --output_dir=$OUTPUT_DIR
```

Note that this command will collect some frames from the real environment for
random starts.

The same command can be used to resume interrupted training - checkpoints are
saved in `output_dir`.

We use `NoFrameskip-v4` game mode with our own frame skip (4 by default).

The training script runs periodic evaluation, but with timestep limit 1000 to
make it faster. To do full evaluation after training, run:

```
python -m tensor2tensor.rl.evaluator \
  --loop_hparams_set=rlmb_base_stochastic_discrete \
  --hparams=game=pong \
  --policy_dir=$OUTPUT_DIR \
  --eval_metrics_dir=$OUTPUT_DIR/full_eval_metrics
```


## Full model-based training

Our full training pipeline involves alternating between collecting data using
policy, training the world model and training the policy inside the model. It
requires significantly more time (several days to a week, depending on your
hardware and the model you use).

To train a deterministic model:

```
python -m tensor2tensor.rl.trainer_model_based \
  --loop_hparams_set=rlmb_base \
  --loop_hparams=game=pong \
  --output_dir ~/t2t_train/mb_det_pong
```

To train a stochastic discrete model:

```
python -m tensor2tensor.rl.trainer_model_based \
  --loop_hparams_set=rlmb_base_stochastic_discrete \
  --loop_hparams=game=pong \
  --output_dir ~/t2t_train/mb_sd_pong
```

Hyperparameter sets are defined in
`tensor2tensor/rl/trainer_model_based_params.py`. Hyperparameter sets for the
world model and agent are nested within `loop_hparams` by name. You can change
them with:

```
  --loop_hparams=game=freeway,generative_model=next_frame_basic_deterministic,base_algo_params=ppo_original_params
```

Game names should be provided in `snake_case`.


## Using checkpoints for other games

We provide pretrained policies and stochastic discrete models for most of the
Atari games in OpenAI Gym. They are available in Google Cloud Storage at
`gs://tensor2tensor-checkpoints/modelrl_experiments/train_sd/N`, where `N` is
a run number in range 1 - 180. Games with checkpoints are defined in
`tensor2tensor.data_generators.gym_env.ATARI_GAMES_WITH_HUMAN_SCORE_NICE` and
are numbered according to this order, with 5 runs per game. For example, runs
for Amidar have numbers 6 - 10.
