# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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
from tensor2tensor.utils import hparam
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


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


def _rlmb_base():
  return hparam.HParams(
      epochs=15,
      # Total frames used for training. This will be distributed evenly across
      # hparams.epochs.
      # This number should be divisible by real_ppo_epoch_length*epochs
      # for our frame accounting to be preceise.
      num_real_env_frames=96000,
      generative_model="next_frame_basic_deterministic",
      generative_model_params="next_frame_pixel_noise",
      autoencoder_train_steps=0,
      autoencoder_train_steps_initial_multiplier=10,
      autoencoder_hparams_set="autoencoder_discrete_pong",
      model_train_steps=15000,
      initial_epoch_train_steps_multiplier=3,
      # Use random starts when learning agent on simulated env.
      simulation_random_starts=True,
      # Flip the first random frame in PPO batch for the true beginning.
      simulation_flip_first_random_for_beginning=False,
      intrinsic_reward_scale=0.,
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

      # Policy sampling temperature to use when gathering data from the real
      # environment.
      real_sampling_temp=1.0,

      # Sampling temperatures to try during eval.
      eval_sampling_temps=[0.5, 0.0, 1.0],
      eval_max_num_noops=8,
      # To speed up the pipeline. Some games want to run forever.
      eval_rl_env_max_episode_steps=1000,

      game="pong",
      sticky_actions=False,
      # If set, use this as the gym env name, instead of changing game mode etc.
      rl_env_name="",
      # Controls whether we should derive observation space, do some
      # pre-processing etc. See T2TGymEnv._derive_observation_space.
      rl_should_derive_observation_space=True,
      # Whether to evaluate the world model in each iteration of the loop to get
      # the model_reward_accuracy metric.
      eval_world_model=True,
      # Number of concurrent rollouts in world model evaluation.
      wm_eval_batch_size=16,
      # Number of batches to run for world model evaluation.
      wm_eval_num_batches=8,
      # Ratios of ppo_epoch_length to report reward_accuracy on.
      wm_eval_rollout_ratios=[0.25, 1],
      stop_loop_early=False,  # To speed-up tests.
      rl_env_max_episode_steps=-1,  # Use default from gym.make()
      # Number of last observations to feed to the agent and world model.
      frame_stack_size=4,
      # This is only used for world-model evaluation currently, PolicyLearner
      # uses algorithm specific hparams to set this during training.
      simulated_rollout_length=50,
      wm_policy_param_sharing=False,

      # To be overridden.
      base_algo="",
      base_algo_params="",
      # Number of real environments to train on simultaneously.
      real_batch_size=-1,
      # Number of simulated environments to train on simultaneously.
      simulated_batch_size=-1,
      # Batch size during evaluation. Metrics are averaged over this number of
      # rollouts.
      eval_batch_size=-1,
  )


def update_hparams(hparams, other):
  for key, value in six.iteritems(other):
    if key in hparams.values():
      hparams.set_hparam(key, value)
    else:
      hparams.add_hparam(key, value)


@registry.register_hparams
def rlmb_ppo_base():
  """HParams for PPO base."""
  hparams = _rlmb_base()
  ppo_params = dict(
      base_algo="ppo",
      base_algo_params="ppo_original_params",
      # Number of real environments to train on simultaneously.
      real_batch_size=1,
      # Number of simulated environments to train on simultaneously.
      simulated_batch_size=16,
      eval_batch_size=32,

      # Unused; number of PPO epochs is calculated from the real frame limit.
      real_ppo_epochs_num=0,
      # Number of frames that can be taken from the simulated environment before
      # it diverges, used for training the agent.

      ppo_epochs_num=1000,  # This should be enough to see something
      # Should be equal to simulated_rollout_length.
      # TODO(koz4k): Uncouple this by outputing done from SimulatedBatchEnv.
      ppo_epoch_length=hparams.simulated_rollout_length,
      # Do not eval since simulated batch env does not produce dones
      ppo_eval_every_epochs=0,
      ppo_learning_rate_constant=1e-4,  # Will be changed, just so it exists.
      # This needs to be divisible by real_ppo_effective_num_agents.
      real_ppo_epoch_length=16 * 200,
      real_ppo_learning_rate_constant=1e-4,
      real_ppo_effective_num_agents=16,
      real_ppo_eval_every_epochs=0,

      simulation_flip_first_random_for_beginning=True,
  )
  update_hparams(hparams, ppo_params)
  return hparams


@registry.register_hparams
def rlmb_ppo_base_param_sharing():
  """HParams for PPO base with parameter sharing."""
  hparams = rlmb_ppo_base()
  hparams.wm_policy_param_sharing = True
  hparams.base_algo_params = "ppo_original_world_model"
  return hparams


@registry.register_hparams
def rlmb_base():
  return rlmb_ppo_base()


@registry.register_hparams
def rlmb_dqn_base():
  """rlmb_dqn_base params."""
  hparams = _rlmb_base()
  simulated_rollout_length = 10
  dqn_params = dict(
      base_algo="dqn",
      base_algo_params="dqn_original_params",
      real_batch_size=1,
      simulated_batch_size=16,
      dqn_agent_generates_trainable_dones=False,
      eval_batch_size=1,
      # Must be equal to dqn_time_limit for now
      simulated_rollout_length=simulated_rollout_length,
      dqn_time_limit=simulated_rollout_length,
      simulation_flip_first_random_for_beginning=False,
      dqn_eval_episodes_num=3,

      # TODO(kc): only for model-free compatibility, remove this
      epochs_num=-1,
  )
  update_hparams(hparams, dqn_params)
  return hparams


@registry.register_hparams
def rlmb_dqn_guess1():
  """DQN guess1 params."""
  hparams = rlmb_dqn_base()
  hparams.set_hparam("base_algo_params", "dqn_guess1_params")
  # At the moment no other option for evaluation, so we want long rollouts to
  # not bias scores.
  hparams.set_hparam("eval_rl_env_max_episode_steps", 5000)
  return hparams


@registry.register_hparams
def rlmb_dqn_guess1_rainbow():
  """Rainbow rlmb_dqn guess1 params."""
  hparams = rlmb_dqn_guess1()
  hparams.set_hparam("base_algo_params", "dqn_guess1_rainbow_params")
  return hparams


@registry.register_hparams
def rlmb_dqn_rainbow_large_epsilon():
  """Rainbow rlmb_dqn params."""
  hparams = rlmb_dqn_guess1()
  hparams.set_hparam("base_algo_params", "dqn_rainbow_params")
  hparams.set_hparam("dqn_agent_epsilon_train", 0.1)
  hparams.add_hparam("real_dqn_agent_epsilon_train", 0.02)
  simulated_rollout_length = 10
  hparams.set_hparam("simulated_rollout_length", simulated_rollout_length)
  hparams.set_hparam("dqn_time_limit", simulated_rollout_length)
  return hparams


@registry.register_hparams
def rlmb_dqn_guess1_2m_replay_buffer():
  """DQN guess1 params, 2M replay buffer."""
  hparams = rlmb_dqn_guess1()
  hparams.set_hparam("base_algo_params", "dqn_2m_replay_buffer_params")
  return hparams


@registry.register_hparams
def rlmb_dqn_guess1_10m_replay_buffer():
  """DQN guess1 params, 10M replay buffer."""
  hparams = rlmb_dqn_guess1()
  hparams.set_hparam("base_algo_params", "dqn_10m_replay_buffer_params")
  return hparams


@registry.register_hparams
def rlmb_basetest():
  """Base setting but quicker with only 2 epochs."""
  hparams = rlmb_base()
  hparams.game = "pong"
  hparams.epochs = 2
  hparams.num_real_env_frames = 3200
  hparams.model_train_steps = 100
  hparams.ppo_epochs_num = 2
  return hparams


@registry.register_hparams
def rlmb_noresize():
  hparams = rlmb_base()
  hparams.resize_height_factor = 1
  hparams.resize_width_factor = 1
  return hparams


@registry.register_hparams
def rlmb_ppo_quick():
  """Base setting but quicker with only 2 epochs."""
  hparams = rlmb_ppo_base()
  hparams.epochs = 2
  hparams.model_train_steps = 25000
  hparams.ppo_epochs_num = 700
  hparams.ppo_epoch_length = 50
  return hparams


@registry.register_hparams
def rlmb_quick():
  """Base setting but quicker with only 2 epochs."""
  return rlmb_ppo_quick()


@registry.register_hparams
def rlmb_ppo_quick_param_sharing():
  """HParams for PPO quick with parameter sharing."""
  hparams = rlmb_ppo_quick()
  hparams.wm_policy_param_sharing = True
  hparams.base_algo_params = "ppo_original_world_model"
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
  # The parameters below are the same as base, but repeated for easier reading.
  hparams.ppo_epoch_length = 50
  hparams.simulated_rollout_length = 50
  hparams.simulated_batch_size = 16
  return hparams


@registry.register_hparams
def rlmb_base_stochastic_discrete_sticky_actions():
  """Base setting, stochastic discrete model with sticky action environment."""
  hparams = rlmb_base_stochastic_discrete()
  hparams.sticky_actions = True
  return hparams


@registry.register_hparams
def rlmb_base_stochastic_discrete_20k():
  """Base setting with stochastic discrete model with 20k steps."""
  hparams = rlmb_base_stochastic_discrete()
  # Our num_real_env_frames should be divisible by real_ppo_epoch_length*epochs
  # Here we decrease epochs to 6 and make this number 16*200*6.
  hparams.num_real_env_frames = 19200
  hparams.epochs = 6
  hparams.ppo_epochs_num = 2000  # Increase PPO steps as we have less epochs.
  return hparams


@registry.register_hparams
def rlmb_base_stochastic_discrete_50k():
  """Base setting with stochastic discrete model with 50k steps."""
  hparams = rlmb_base_stochastic_discrete()
  hparams.num_real_env_frames = 48000
  return hparams


@registry.register_hparams
def rlmb_base_stochastic_discrete_75k_model_steps():
  """Base setting with stochastic discrete model with 75k WM steps."""
  hparams = rlmb_base_stochastic_discrete()
  hparams.model_train_steps = 15000 * 5
  return hparams


@registry.register_hparams
def rlmb_base_stochastic_discrete_20k_model_steps():
  """Base SD setting with 20k WM steps."""
  hparams = rlmb_base_stochastic_discrete()
  hparams.model_train_steps = 20000
  return hparams


@registry.register_hparams
def rlmb_base_stochastic_discrete_30k_model_steps():
  """Base SD setting with 20k WM steps."""
  hparams = rlmb_base_stochastic_discrete()
  hparams.model_train_steps = 30000
  return hparams


@registry.register_hparams
def rlmb_base_stochastic_discrete_200k():
  """Base setting with stochastic discrete model with 200k steps."""
  hparams = rlmb_base_stochastic_discrete()
  hparams.num_real_env_frames = 96000 * 2
  return hparams


@registry.register_hparams
def rlmb_base_stochastic_discrete_500k():
  """Base setting with stochastic discrete model with 500k steps."""
  hparams = rlmb_base_stochastic_discrete()
  hparams.num_real_env_frames = 96000 * 5
  return hparams


@registry.register_hparams
def rlmb_base_stochastic_discrete_1m():
  """Base setting with stochastic discrete model with 1M steps."""
  hparams = rlmb_base_stochastic_discrete()
  hparams.num_real_env_frames = 96000 * 10
  return hparams


@registry.register_hparams
def rlmb_base_stochastic_discrete_param_sharing():
  """Base setting with stochastic discrete model with parameter sharing."""
  hparams = rlmb_base_stochastic_discrete()
  hparams.wm_policy_param_sharing = True
  hparams.base_algo_params = "ppo_original_world_model_stochastic_discrete"
  return hparams


@registry.register_hparams
def rlmb_long():
  """Long setting with base model."""
  hparams = rlmb_base()
  hparams.generative_model_params = "next_frame_pixel_noise_long"
  return hparams


@registry.register_hparams
def rlmb_long_stochastic_discrete():
  """Long setting with stochastic discrete model."""
  hparams = rlmb_base_stochastic_discrete()
  hparams.generative_model_params = "next_frame_basic_stochastic_discrete_long"
  hparams.ppo_epochs_num = 1000
  return hparams


@registry.register_hparams
def rlmb_long_stochastic_discrete_planner():
  hparams = rlmb_long_stochastic_discrete()
  hparams.eval_batch_size = 1
  hparams.eval_sampling_temps = [3.0]
  hparams.eval_max_num_noops = 0
  return hparams


@registry.register_hparams
def rlmb_long_stochastic_discrete_simulation_deterministic_starts():
  """Long setting with stochastic discrete model & deterministic sim starts."""
  hparams = rlmb_base_stochastic_discrete()
  hparams.generative_model_params = "next_frame_basic_stochastic_discrete_long"
  hparams.ppo_epochs_num = 1000
  hparams.simulation_random_starts = False
  return hparams


@registry.register_hparams
def rlmb_long_stochastic_discrete_100steps():
  """Long setting with stochastic discrete model, changed ppo steps."""
  hparams = rlmb_long_stochastic_discrete()
  hparams.ppo_epoch_length = 100
  hparams.simulated_rollout_length = 100
  hparams.simulated_batch_size = 8
  return hparams


@registry.register_hparams
def rlmb_long_stochastic_discrete_25steps():
  """Long setting with stochastic discrete model, changed ppo steps."""
  hparams = rlmb_long_stochastic_discrete()
  hparams.ppo_epoch_length = 25
  hparams.simulated_rollout_length = 25
  hparams.simulated_batch_size = 32
  return hparams


@registry.register_hparams
def rlmb_long_stochastic_discrete_gamma95():
  """Long setting with stochastic discrete model, changed gamma."""
  hparams = rlmb_long_stochastic_discrete()
  hparams.base_algo_params = "ppo_original_params_gamma95"
  return hparams


@registry.register_hparams
def rlmb_long_stochastic_discrete_gamma90():
  """Long setting with stochastic discrete model, changed gamma."""
  hparams = rlmb_long_stochastic_discrete()
  hparams.base_algo_params = "ppo_original_params_gamma90"
  return hparams


@registry.register_hparams
def rlmb_base_stochastic_discrete_3epochs():
  """Long setting with stochastic discrete model, changed epochs."""
  hparams = rlmb_base_stochastic_discrete()
  hparams.epochs = 3
  hparams.ppo_epochs_num = 2000
  return hparams


@registry.register_hparams
def rlmb_base_stochastic_discrete_1epoch():
  """Long setting with stochastic discrete model, changed epochs."""
  hparams = rlmb_base_stochastic_discrete()
  hparams.epochs = 1
  hparams.ppo_epochs_num = 3000
  return hparams


@registry.register_hparams
def rlmb_base_recurrent():
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


def _rlmb_tiny_overrides():
  """Parameters to override for tiny setting excluding agent-related hparams."""
  return dict(
      epochs=1,
      num_real_env_frames=128,
      model_train_steps=2,
      max_num_noops=1,
      eval_max_num_noops=1,
      generative_model_params="next_frame_tiny",
      stop_loop_early=True,
      resize_height_factor=2,
      resize_width_factor=2,
      wm_eval_rollout_ratios=[1],
      rl_env_max_episode_steps=7,
      eval_rl_env_max_episode_steps=7,
      simulated_rollout_length=2,
      eval_sampling_temps=[0.0, 1.0],
  )


@registry.register_hparams
def rlmb_ppo_tiny():
  """Tiny set for testing."""
  hparams = rlmb_ppo_base()
  hparams = hparams.override_from_dict(_rlmb_tiny_overrides())
  update_hparams(hparams, dict(
      ppo_epochs_num=2,
      ppo_epoch_length=10,
      real_ppo_epoch_length=36,
      real_ppo_effective_num_agents=2,
      real_batch_size=1,
      eval_batch_size=1,
  ))
  return hparams


@registry.register_hparams
def rlmb_tiny():
  return rlmb_ppo_tiny()


@registry.register_hparams
def rlmb_dqn_tiny():
  """Tiny set for testing."""
  hparams = rlmb_dqn_base()
  hparams = hparams.override_from_dict(_rlmb_tiny_overrides())
  update_hparams(hparams, dict(
      base_algo_params="dqn_guess1_params",
      simulated_rollout_length=2,
      dqn_time_limit=2,
      dqn_num_frames=128,
      real_dqn_replay_buffer_replay_capacity=100,
      dqn_replay_buffer_replay_capacity=100,
      real_dqn_agent_min_replay_history=10,
      dqn_agent_min_replay_history=10,
  ))
  return hparams


@registry.register_hparams
def rlmb_tiny_stochastic():
  """Tiny setting with a stochastic next-frame model."""
  hparams = rlmb_ppo_tiny()
  hparams.epochs = 1  # Too slow with 2 for regular runs.
  hparams.generative_model = "next_frame_basic_stochastic"
  hparams.generative_model_params = "next_frame_basic_stochastic"
  return hparams


@registry.register_hparams
def rlmb_tiny_recurrent():
  """Tiny setting with a recurrent next-frame model."""
  hparams = rlmb_ppo_tiny()
  hparams.epochs = 1  # Too slow with 2 for regular runs.
  hparams.generative_model = "next_frame_basic_recurrent"
  hparams.generative_model_params = "next_frame_basic_recurrent"
  return hparams


@registry.register_hparams
def rlmb_tiny_sv2p():
  """Tiny setting with a tiny sv2p model."""
  hparams = rlmb_ppo_tiny()
  hparams.generative_model = "next_frame_sv2p"
  hparams.generative_model_params = "next_frame_sv2p_tiny"
  hparams.grayscale = False
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
  rhp.set_discrete("loop.ppo_learning_rate_constant", [5e-5, 1e-4, 2e-4])
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
  rhp.set_categorical("loop.game",
                      gym_env.ATARI_GAMES_WITH_HUMAN_SCORE_NICE)
  rhp.set_discrete("model.moe_loss_coef", list(range(5)))


@registry.register_ranged_hparams
def rlmb_human_score_games_v100unfriendly(rhp):
  """Games that for strange reasons often fail on v100s but work on p100s."""
  rhp.set_categorical("loop.game",
                      ["chopper_command", "boxing", "asterix", "seaquest"])
  rhp.set_discrete("model.moe_loss_coef", list(range(5)))


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
  rhp.set_float("loop.ppo_learning_rate_constant", base_lr / 2, base_lr * 2)


@registry.register_ranged_hparams
def rlmb_ppolr(rhp):
  base_lr = 1e-4
  rhp.set_float("loop.ppo_learning_rate_constant", base_lr / 2, base_lr * 2)


@registry.register_ranged_hparams
def rlmb_ae_ppo_lr(rhp):
  rhp.set_categorical("loop.game", ["breakout", "pong", "freeway"])
  base_lr = 1e-4
  rhp.set_float("loop.ppo_learning_rate_constant", base_lr / 2, base_lr * 2)


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
  rhp.set_discrete("ppo.logits_clip", [0., 4.0])


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

  return hparam.HParams(**merged_values)


def split_scoped_hparams(scopes, merged_hparams):
  """Split single HParams with scoped keys into multiple."""
  split_values = {scope: {} for scope in scopes}
  merged_values = merged_hparams.values()
  for scoped_key, value in six.iteritems(merged_values):
    scope = scoped_key.split(".")[0]
    key = scoped_key[len(scope) + 1:]
    split_values[scope][key] = value

  return [
      hparam.HParams(**split_values[scope]) for scope in scopes
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
    return hparam.HParams(**hparams.values())

  return new_hparams_set


def create_loop_hparams():
  hparams = registry.hparams(FLAGS.loop_hparams_set)
  hparams.parse(FLAGS.loop_hparams)
  return hparams
