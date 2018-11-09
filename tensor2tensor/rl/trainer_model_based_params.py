# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Parameter sets for training of model-based RL agents."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six


from tensor2tensor.data_generators import gym_env
from tensor2tensor.utils import registry

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string("loop_hparams_set", "rlmb_base",
                    "Which RL hparams set to use.")
flags.DEFINE_string("loop_hparams", "", "Overrides for overall loop HParams.")
flags.DEFINE_string("job_dir_to_evaluate", "",
                    "Directory of a job to be evaluated.")
flags.DEFINE_string("eval_results_dir", "/tmp",
                    "Directory to store result of evaluation")


HP_SCOPES = ["loop", "model", "ppo"]


@registry.register_hparams
def rlmb_base():
  return tf.contrib.training.HParams(
      epochs=15,
      # Total frames used for training. This will be distributed evenly across
      # hparams.epochs.
      # This number should be divisible by real_ppo_epoch_length*epochs
      # for our frame accounting to be preceise.
      num_real_env_frames=96000,
      generative_model="next_frame_basic_deterministic",
      generative_model_params="next_frame_pixel_noise",
      base_algo="ppo",
      base_algo_params="ppo_original_params",
      autoencoder_train_steps=0,
      autoencoder_train_steps_initial_multiplier=10,
      autoencoder_hparams_set="autoencoder_discrete_pong",
      model_train_steps=15000,
      initial_epoch_train_steps_multiplier=3,
      simulated_env_generator_num_steps=2000,
      simulation_random_starts=True,  # Use random starts in PPO.
      # Flip the first random frame in PPO batch for the true beginning.
      simulation_flip_first_random_for_beginning=True,
      intrinsic_reward_scale=0.,
      ppo_epochs_num=1000,  # This should be enough to see something
      # Our simulated envs do not know how to reset.
      # You should set ppo_time_limit to the value you believe that
      # the simulated env produces a reasonable output.
      ppo_time_limit=200,  # TODO(blazej): this param is unused
      # It makes sense to have ppo_time_limit=ppo_epoch_length,
      # though it is not necessary.
      ppo_epoch_length=50,
      ppo_num_agents=16,
      # Do not eval since simulated batch env does not produce dones
      ppo_eval_every_epochs=0,
      ppo_learning_rate=1e-4,  # Will be changed, just so it exists.
      # Whether the PPO agent should be restored from the previous iteration, or
      # should start fresh each time.
      ppo_continue_training=True,
      # Resizing.
      resize_height_factor=2,
      resize_width_factor=2,
      grayscale=False,
      # Maximum number of noops to make on environment reset.
      max_num_noops=8,
      # Bump learning rate after first epoch by 3x.
      # We picked 3x because our default learning rate schedule decreases with
      # 1/square root of step; 1/sqrt(10k) = 0.01 and 1/sqrt(100k) ~ 0.0032
      # so by bumping it up 3x we about "go back" from 100k steps to 10k, which
      # is approximately as much as "going back 1 epoch" would be.
      # In your experiments, you want to optimize this rate to your schedule.
      learning_rate_bump=3.0,

      real_ppo_epochs_num=0,
      # This needs to be divisible by real_ppo_effective_num_agents.
      real_ppo_epoch_length=16*200,
      real_ppo_num_agents=1,
      real_ppo_learning_rate=1e-4,
      real_ppo_continue_training=True,
      real_ppo_effective_num_agents=16,
      real_ppo_eval_every_epochs=0,

      eval_num_agents=30,
      eval_max_num_noops=8,

      game="pong",
      # Whether to evaluate the world model in each iteration of the loop to get
      # the model_reward_accuracy metric.
      eval_world_model=True,
      # Number of concurrent rollouts in world model evaluation.
      wm_eval_batch_size=16,
      # Number of batches to run for world model evaluation.
      wm_eval_epochs_num=8,
      # Ratios of ppo_epoch_length to report reward_accuracy on.
      wm_eval_rollout_ratios=[0.25, 0.5, 1, 2],
      stop_loop_early=False,  # To speed-up tests.
      env_timesteps_limit=-1,  # Use default from gym.make()
      # Number of last observations to feed to the agent and world model.
      frame_stack_size=4,
  )


@registry.register_hparams
def rlmb_basetest():
  """Base setting but quicker with only 2 epochs."""
  hparams = rlmb_base()
  hparams.game = "pong"
  hparams.epochs = 2
  hparams.num_real_env_frames = 3200
  hparams.model_train_steps = 100
  hparams.simulated_env_generator_num_steps = 20
  hparams.ppo_epochs_num = 2
  return hparams


@registry.register_hparams
def rlmb_noresize():
  hparams = rlmb_base()
  hparams.resize_height_factor = 1
  hparams.resize_width_factor = 1
  return hparams


@registry.register_hparams
def rlmb_quick():
  """Base setting but quicker with only 2 epochs."""
  hparams = rlmb_base()
  hparams.epochs = 2
  hparams.model_train_steps = 25000
  hparams.ppo_epochs_num = 700
  hparams.ppo_epoch_length = 50
  return hparams


@registry.register_hparams
def rlmb_quick_noresize():
  hparams = rlmb_base()
  hparams.resize_height_factor = 1
  hparams.resize_width_factor = 1
  return hparams


@registry.register_hparams
def rlmb_quick_sd():
  """Quick setting with stochastic discrete model."""
  hparams = rlmb_quick()
  hparams.generative_model = "next_frame_basic_stochastic_discrete"
  hparams.generative_model_params = "next_frame_basic_stochastic_discrete"
  return hparams


@registry.register_hparams
def rlmb_sdtest():
  """Test setting with stochastic discrete model."""
  hparams = rlmb_basetest()
  hparams.generative_model = "next_frame_basic_stochastic_discrete"
  hparams.generative_model_params = "next_frame_basic_stochastic_discrete"
  return hparams


@registry.register_hparams
def rlmb_quick_sm():
  """Quick setting with sampling."""
  hparams = rlmb_quick()
  hparams.generative_model_params = "next_frame_sampling"
  return hparams


@registry.register_hparams
def rlmb_base_stochastic():
  """Base setting with a stochastic next-frame model."""
  hparams = rlmb_base()
  hparams.initial_epoch_train_steps_multiplier = 5
  hparams.generative_model = "next_frame_basic_stochastic"
  hparams.generative_model_params = "next_frame_basic_stochastic"
  return hparams


@registry.register_hparams
def rlmb_base_sampling_stochastic():
  """Base setting with a stochastic next-frame model."""
  hparams = rlmb_base()
  hparams.generative_model = "next_frame_basic_stochastic"
  hparams.generative_model_params = "next_frame_sampling_stochastic"
  return hparams


@registry.register_hparams
def rlmb_base_stochastic_discrete():
  """Base setting with stochastic discrete model."""
  hparams = rlmb_base()
  hparams.learning_rate_bump = 1.0
  hparams.grayscale = False
  hparams.generative_model = "next_frame_basic_stochastic_discrete"
  hparams.generative_model_params = "next_frame_basic_stochastic_discrete"
  return hparams


@registry.register_hparams
def rlmb_base_stochastic_recurrent():
  """Base setting with recurrent model."""
  hparams = rlmb_base()
  hparams.generative_model = "next_frame_basic_recurrent"
  hparams.generative_model_params = "next_frame_basic_recurrent"
  return hparams


@registry.register_hparams
def rlmb_base_stochastic_discrete_noresize():
  """Base setting with stochastic discrete model."""
  hparams = rlmb_base()
  hparams.generative_model = "next_frame_basic_stochastic_discrete"
  hparams.generative_model_params = "next_frame_basic_stochastic_discrete"
  hparams.resize_height_factor = 1
  hparams.resize_width_factor = 1
  return hparams


@registry.register_hparams
def rlmb_base_sv2p():
  """Base setting with sv2p as world model."""
  hparams = rlmb_base()
  hparams.learning_rate_bump = 1.0
  hparams.generative_model = "next_frame_sv2p"
  hparams.generative_model_params = "next_frame_sv2p_atari"
  return hparams


@registry.register_hparams
def rlmb_base_sv2p_softmax():
  """Base setting with sv2p as world model with softmax."""
  hparams = rlmb_base_sv2p()
  hparams.generative_model_params = "next_frame_sv2p_atari_softmax"
  return hparams


@registry.register_hparams
def rlmb_base_sv2p_deterministic():
  """Base setting with deterministic sv2p as world model."""
  hparams = rlmb_base_sv2p()
  hparams.generative_model_params = "next_frame_sv2p_atari_deterministic"
  return hparams


@registry.register_hparams
def rlmb_base_sv2p_deterministic_softmax():
  """Base setting with deterministic sv2p as world model with softmax."""
  hparams = rlmb_base_sv2p_softmax()
  hparams.generative_model_params = (
      "next_frame_sv2p_atari_softmax_deterministic")
  return hparams


@registry.register_hparams
def rlmb_base_sv2p_flippy30():
  """Base setting with sv2p as world model."""
  hparams = rlmb_base()
  hparams.epochs = 30
  hparams.ppo_epochs_num = 1000
  hparams.model_train_steps = 15000
  hparams.learning_rate_bump = 1.0
  hparams.initial_epoch_train_steps_multiplier = 5
  hparams.generative_model = "next_frame_sv2p"
  hparams.generative_model_params = "next_frame_sv2p_atari"
  return hparams


@registry.register_hparams
def rlmb_base_sv2p_softmax_flippy30():
  """Base setting with sv2p as world model with softmax."""
  hparams = rlmb_base_sv2p_flippy30()
  hparams.generative_model_params = "next_frame_sv2p_atari_softmax"
  return hparams


@registry.register_hparams
def rlmb_base_sv2p_deterministic_flippy30():
  """Base setting with deterministic sv2p as world model."""
  hparams = rlmb_base_sv2p_flippy30()
  hparams.generative_model_params = "next_frame_sv2p_atari_deterministic"
  return hparams


@registry.register_hparams
def rlmb_base_sv2p_deterministic_softmax_flippy30():
  """Base setting with deterministic sv2p as world model with softmax."""
  hparams = rlmb_base_sv2p_softmax_flippy30()
  hparams.generative_model_params = (
      "next_frame_sv2p_atari_softmax_deterministic")
  return hparams


@registry.register_hparams
def rlmb_base_sampling():
  """Base setting with a stochastic next-frame model."""
  hparams = rlmb_base()
  hparams.generative_model_params = "next_frame_sampling"
  return hparams


@registry.register_hparams
def rlmb_base_sampling_noresize():
  hparams = rlmb_base_sampling()
  hparams.resize_height_factor = 1
  hparams.resize_width_factor = 1
  return hparams


@registry.register_hparams
def rlmb_flippy60():
  """Schedule with a lot of epochs (slow)."""
  hparams = rlmb_base_sampling()
  hparams.epochs = 60
  hparams.ppo_epochs_num = 500
  hparams.model_train_steps = 10000
  return hparams


@registry.register_hparams
def rlmb_flippy30():
  """Schedule with a lot of epochs (slow)."""
  hparams = rlmb_base_sampling()
  hparams.epochs = 30
  hparams.ppo_epochs_num = 1000
  hparams.model_train_steps = 15000
  return hparams


@registry.register_hparams
def rlmb_medium():
  """Small set for larger testing."""
  hparams = rlmb_base()
  hparams.num_real_env_frames //= 2
  return hparams


@registry.register_hparams
def rlmb_25k():
  """Small set for larger testing."""
  hparams = rlmb_medium()
  hparams.num_real_env_frames //= 2
  return hparams


@registry.register_hparams
def rlmb_short():
  """Small set for larger testing."""
  hparams = rlmb_base()
  hparams.num_real_env_frames //= 5
  hparams.model_train_steps //= 10
  hparams.ppo_epochs_num //= 10
  return hparams


@registry.register_hparams
def rlmb_model_only():
  hp = rlmb_base()
  hp.epochs = 1
  hp.ppo_epochs_num = 0
  return hp


@registry.register_hparams
def rlmb_tiny():
  """Tiny set for testing."""
  return rlmb_base_sampling().override_from_dict(
      tf.contrib.training.HParams(
          epochs=1,
          num_real_env_frames=128,
          simulated_env_generator_num_steps=64,
          model_train_steps=2,
          ppo_epochs_num=2,
          ppo_time_limit=5,
          ppo_epoch_length=2,
          ppo_num_agents=2,
          real_ppo_epoch_length=36,
          real_ppo_num_agents=1,
          real_ppo_epochs_num=0,
          real_ppo_effective_num_agents=2,
          eval_num_agents=1,
          generative_model_params="next_frame_tiny",
          stop_loop_early=True,
          resize_height_factor=2,
          resize_width_factor=2,
          game="pong",
          wm_eval_rollout_ratios=[1],
          env_timesteps_limit=6,
      ).values())


@registry.register_hparams
def rlmb_tiny_stochastic():
  """Tiny setting with a stochastic next-frame model."""
  hparams = rlmb_tiny()
  hparams.epochs = 1  # Too slow with 2 for regular runs.
  hparams.generative_model = "next_frame_basic_stochastic"
  hparams.generative_model_params = "next_frame_basic_stochastic"
  return hparams


@registry.register_hparams
def rlmb_tiny_recurrent():
  """Tiny setting with a recurrent next-frame model."""
  hparams = rlmb_tiny()
  hparams.epochs = 1  # Too slow with 2 for regular runs.
  hparams.generative_model = "next_frame_basic_recurrent"
  hparams.generative_model_params = "next_frame_basic_recurrent"
  return hparams


@registry.register_hparams
def rlmb_tiny_sv2p():
  """Tiny setting with a tiny sv2p model."""
  hparams = rlmb_tiny()
  hparams.generative_model = "next_frame_sv2p"
  hparams.generative_model_params = "next_frame_sv2p_tiny"
  hparams.grayscale = False
  return hparams


@registry.register_hparams
def rlmb_ae_base():
  """Parameter set for autoencoders."""
  hparams = rlmb_base()
  hparams.ppo_params = "ppo_pong_ae_base"
  hparams.generative_model_params = "next_frame_ae"
  hparams.autoencoder_hparams_set = "autoencoder_discrete_pong"
  hparams.autoencoder_train_steps = 5000
  hparams.resize_height_factor = 1
  hparams.resize_width_factor = 1
  hparams.grayscale = False
  return hparams


@registry.register_hparams
def rlmb_ae_basetest():
  """Base AE setting but quicker with only 2 epochs."""
  hparams = rlmb_ae_base()
  hparams.game = "pong"
  hparams.epochs = 2
  hparams.num_real_env_frames = 3200
  hparams.model_train_steps = 100
  hparams.autoencoder_train_steps = 10
  hparams.simulated_env_generator_num_steps = 20
  hparams.ppo_epochs_num = 2
  return hparams


@registry.register_hparams
def rlmb_ae_tiny():
  """Tiny set for testing autoencoders."""
  hparams = rlmb_tiny()
  hparams.ppo_params = "ppo_pong_ae_base"
  hparams.generative_model_params = "next_frame_ae_tiny"
  hparams.autoencoder_hparams_set = "autoencoder_discrete_tiny"
  hparams.resize_height_factor = 1
  hparams.resize_width_factor = 1
  hparams.grayscale = False
  hparams.autoencoder_train_steps = 1
  hparams.autoencoder_train_steps_initial_multiplier = 0
  return hparams


@registry.register_hparams
def rlmb_tiny_simulation_deterministic_starts():
  hp = rlmb_tiny()
  hp.simulation_random_starts = False
  return hp


# RangedHParams for tuning
# ==============================================================================
# Note that the items here must be scoped with one of
# HP_SCOPES={loop, model, ppo}, which set hyperparameters for the top-level
# hparams, hp.generative_model_params, and hp.ppo_params, respectively.
@registry.register_ranged_hparams
def rlmb_grid(rhp):
  """Grid over games and frames, and 5 runs each for variance."""
  rhp.set_categorical("loop.game", ["breakout", "pong", "freeway"])
  base = 100000
  medium = base // 2
  small = medium // 2
  rhp.set_discrete("loop.num_real_env_frames", [base, medium, small])

  # Dummy parameter to get 5 runs for each configuration
  rhp.set_discrete("model.moe_loss_coef", list(range(5)))


@registry.register_ranged_hparams
def rlmb_variance(rhp):
  # Dummy parameter to get 5 runs for each configuration
  rhp.set_discrete("model.moe_loss_coef", list(range(5)))
  rhp.set_categorical("loop.game", ["breakout", "pong", "freeway"])


@registry.register_ranged_hparams
def rlmb_variance_nogame(rhp):
  # Dummy parameter to get 20 runs for current configuration.
  rhp.set_discrete("model.moe_loss_coef", list(range(20)))


@registry.register_ranged_hparams
def rlmb_three(rhp):
  rhp.set_discrete("model.moe_loss_coef", list(range(10)))
  rhp.set_categorical("loop.game", ["breakout", "pong", "boxing"])


@registry.register_ranged_hparams
def rlmb_test1(rhp):
  rhp.set_discrete("model.moe_loss_coef", list(range(10)))
  rhp.set_categorical("loop.game", ["breakout", "pong", "boxing"])
  rhp.set_discrete("loop.ppo_learning_rate", [5e-5, 1e-4, 2e-4])
  rhp.set_discrete("ppo.optimization_batch_size", [20, 40])
  rhp.set_discrete("loop.epochs", [3, 6])


@registry.register_ranged_hparams
def rlmb_scheduled_sampling(rhp):
  rhp.set_float("model.scheduled_sampling_prob", 0.0, 1.0)


@registry.register_ranged_hparams
def rlmb_all_games(rhp):
  rhp.set_discrete("model.moe_loss_coef", list(range(5)))
  rhp.set_categorical("loop.game", gym_env.ATARI_GAMES)


@registry.register_ranged_hparams
def rlmb_whitelisted_games(rhp):
  rhp.set_discrete("model.moe_loss_coef", list(range(10)))
  rhp.set_categorical("loop.game", gym_env.ATARI_WHITELIST_GAMES)


@registry.register_ranged_hparams
def rlmb_human_score_games(rhp):
  rhp.set_discrete("model.moe_loss_coef", list(range(10)))
  rhp.set_categorical("loop.game",
                      gym_env.ATARI_GAMES_WITH_HUMAN_SCORE)


@registry.register_ranged_hparams
def rlmb_curious_games10(rhp):
  rhp.set_discrete("model.moe_loss_coef", list(range(10)))
  rhp.set_categorical("loop.game", gym_env.ATARI_CURIOUS_GAMES)


@registry.register_ranged_hparams
def rlmb_curious_games5(rhp):
  rhp.set_discrete("model.moe_loss_coef", list(range(5)))
  rhp.set_categorical("loop.game", gym_env.ATARI_CURIOUS_GAMES)


@registry.register_ranged_hparams
def rlmb_debug_games(rhp):
  rhp.set_discrete("model.moe_loss_coef", list(range(10)))
  rhp.set_categorical("loop.game", gym_env.ATARI_DEBUG_GAMES)


@registry.register_ranged_hparams
def rlmb_ae_variance(rhp):
  # Dummy parameter to get 5 runs for each configuration
  rhp.set_discrete("model.moe_loss_coef", list(range(5)))
  rhp.set_categorical("loop.game", ["breakout", "pong", "freeway"])
  base = 100000
  small = base // 4
  rhp.set_discrete("loop.num_real_env_frames", [base, small])


@registry.register_ranged_hparams
def rlmb_ppolr_game(rhp):
  rhp.set_categorical("loop.game", ["breakout", "pong", "freeway"])
  base_lr = 1e-4
  rhp.set_float("loop.ppo_learning_rate", base_lr / 2, base_lr * 2)


@registry.register_ranged_hparams
def rlmb_ppolr(rhp):
  base_lr = 1e-4
  rhp.set_float("loop.ppo_learning_rate", base_lr / 2, base_lr * 2)


@registry.register_ranged_hparams
def rlmb_ae_ppo_lr(rhp):
  rhp.set_categorical("loop.game", ["breakout", "pong", "freeway"])
  base_lr = 1e-4
  rhp.set_float("loop.ppo_learning_rate", base_lr / 2, base_lr * 2)


@registry.register_ranged_hparams
def rlmb_dropout_range(rhp):
  rhp.set_float("model.dropout", 0.2, 0.4)


@registry.register_ranged_hparams
def rlmb_intrinsic_reward_scale(rhp):
  rhp.set_float("loop.intrinsic_reward_scale", 0.01, 10.)


@registry.register_ranged_hparams
def rlmb_l1l2cutoff_range(rhp):
  """Loss and loss-cutoff tuning grid."""
  rhp.set_float("model.video_modality_loss_cutoff", 1.4, 3.4)


@registry.register_ranged_hparams
def rlmb_xentcutoff_range(rhp):
  """Cross entropy cutoff tuning grid."""
  rhp.set_float("model.video_modality_loss_cutoff", 0.01, 0.05)


@registry.register_ranged_hparams
def rlmb_pixel_noise(rhp):
  """Input pixel noise tuning grid."""
  rhp.set_categorical("loop.generative_model_params",
                      ["next_frame_pixel_noise"])
  rhp.set_discrete("model.video_modality_input_noise",
                   [0.0025 * i for i in range(200)])


@registry.register_ranged_hparams
def rlmb_dummy_range(rhp):
  """Dummy tuning grid just to get the variance."""
  rhp.set_float("model.moe_loss_coef", 0.01, 0.02)


@registry.register_ranged_hparams
def rlmb_epochs_num(rhp):
  rhp.set_categorical("loop.game", gym_env.ATARI_WHITELIST_GAMES)
  rhp.set_discrete("model.moe_loss_coef", list(range(5)))
  rhp.set_discrete("loop.epochs", [3, 6, 12])


@registry.register_ranged_hparams
def rlmb_ppo_epochs_num(rhp):
  rhp.set_categorical("loop.game", gym_env.ATARI_WHITELIST_GAMES)
  rhp.set_discrete("model.moe_loss_coef", list(range(5)))
  rhp.set_discrete("loop.ppo_epochs_num", [200, 1000, 2000, 4000])


@registry.register_ranged_hparams
def rlmb_ppo_epoch_len(rhp):
  rhp.set_categorical("loop.game", gym_env.ATARI_WHITELIST_GAMES)
  rhp.set_discrete("model.moe_loss_coef", list(range(5)))
  rhp.set_discrete("loop.ppo_epoch_length", [25, 50, 100])


@registry.register_ranged_hparams
def rlmb_num_frames(rhp):
  rhp.set_categorical("loop.game", gym_env.ATARI_WHITELIST_GAMES)
  rhp.set_discrete("model.moe_loss_coef", list(range(5)))
  rhp.set_discrete("loop.num_real_env_frames",
                   [1000*el for el in [30, 100, 500, 1000]])


@registry.register_ranged_hparams
def rlmb_ppo_optimization_batch_size(rhp):
  rhp.set_categorical("loop.game", ["pong", "boxing", "seaquest"])
  rhp.set_discrete("model.moe_loss_coef", list(range(10)))
  rhp.set_discrete("ppo.optimization_batch_size", [4, 10, 20])


@registry.register_ranged_hparams
def rlmb_logits_clip(rhp):
  rhp.set_categorical("loop.game", ["pong", "boxing", "seaquest"])
  rhp.set_discrete("model.moe_loss_coef", list(range(10)))
  rhp.set_discrete("ppo.logits_clip", [0., 5.])


@registry.register_ranged_hparams
def rlmb_games_problematic_for_ppo(rhp):
  games = [
      "alien", "boxing", "breakout", "ms_pacman", "video_pinball",
  ]
  rhp.set_categorical("loop.game", games)
  rhp.set_categorical("loop.base_algo_params", ["ppo_original_params"])
  rhp.set_discrete("model.moe_loss_coef", list(range(10)))
  rhp.set_discrete("ppo.dropout_ppo", [0., 0.1])


@registry.register_ranged_hparams
def rlmf_proportional_epoch_length(rhp):
  rhp.set_discrete("proportional_epoch_length", [10, 20, 50, 100, 200, 400])
  rhp.set_categorical("loop.game", gym_env.ATARI_GAMES_WITH_HUMAN_SCORE)


def merge_unscoped_hparams(scopes_and_hparams):
  """Merge multiple HParams into one with scopes."""
  merged_values = {}
  for (scope, hparams) in scopes_and_hparams:
    for key, value in six.iteritems(hparams.values()):
      scoped_key = "%s.%s" % (scope, key)
      merged_values[scoped_key] = value

  return tf.contrib.training.HParams(**merged_values)


def split_scoped_hparams(scopes, merged_hparams):
  """Split single HParams with scoped keys into multiple."""
  split_values = {scope: {} for scope in scopes}
  merged_values = merged_hparams.values()
  for scoped_key, value in six.iteritems(merged_values):
    scope = scoped_key.split(".")[0]
    key = scoped_key[len(scope) + 1:]
    split_values[scope][key] = value

  return [
      tf.contrib.training.HParams(**split_values[scope]) for scope in scopes
  ]


def training_loop_hparams_from_scoped_overrides(scoped_overrides, trial_id):
  """Create HParams suitable for training loop from scoped HParams.

  Args:
    scoped_overrides: HParams, with keys all scoped by one of HP_SCOPES. These
      parameters are overrides for the base HParams created by
      create_loop_hparams.
    trial_id: str, trial identifier. This is used to register unique HParams
      names for the underlying model and ppo HParams.

  Returns:
    HParams suitable for passing to training_loop.
  """
  trial_hp_overrides = scoped_overrides.values()

  # Create loop, model, and ppo base HParams
  loop_hp = create_loop_hparams()
  model_hp_name = trial_hp_overrides.get(
      "loop.generative_model_params", loop_hp.generative_model_params)
  model_hp = registry.hparams(model_hp_name).parse(FLAGS.hparams)
  base_algo_params_name = trial_hp_overrides.get(
      "loop.base_algo_params", loop_hp.base_algo_params)
  algo_hp = registry.hparams(base_algo_params_name)

  # Merge them and then override with the scoped overrides
  combined_hp = merge_unscoped_hparams(
      zip(HP_SCOPES, [loop_hp, model_hp, algo_hp]))
  combined_hp.override_from_dict(trial_hp_overrides)

  # Split out the component hparams
  loop_hp, model_hp, algo_hp = (
      split_scoped_hparams(HP_SCOPES, combined_hp))

  # Dynamic register the model hp and set the new name in loop_hp
  model_hp_name = "model_hp_%s" % str(trial_id)
  dynamic_register_hparams(model_hp_name, model_hp)
  loop_hp.generative_model_params = model_hp_name

  # Dynamic register the algo hp and set the new name in loop_hp
  algo_hp_name = "algo_hp_%s" % str(trial_id)
  dynamic_register_hparams(algo_hp_name, algo_hp)
  loop_hp.base_algo_params = algo_hp_name

  return loop_hp


def dynamic_register_hparams(name, hparams):

  @registry.register_hparams(name)
  def new_hparams_set():
    return tf.contrib.training.HParams(**hparams.values())

  return new_hparams_set


def create_loop_hparams():
  hparams = registry.hparams(FLAGS.loop_hparams_set)
  hparams.parse(FLAGS.loop_hparams)
  return hparams
