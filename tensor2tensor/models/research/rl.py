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

"""Reinforcement learning models and parameters."""

import collections
import functools
import operator
import gym
import six

from tensor2tensor.data_generators import gym_env
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import video_utils
from tensor2tensor.envs import tic_tac_toe_env
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import discretization
from tensor2tensor.layers import modalities
from tensor2tensor.models.video import basic_deterministic_params
from tensor2tensor.models.video import basic_stochastic
from tensor2tensor.rl.envs.py_func_batch_env import PyFuncBatchEnv
from tensor2tensor.rl.envs.simulated_batch_env import SimulatedBatchEnv
from tensor2tensor.rl.envs.simulated_batch_gym_env import SimulatedBatchGymEnv
from tensor2tensor.utils import hparam
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import trainer_lib

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


@registry.register_hparams
def ppo_base_v1():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.learning_rate_schedule = "constant"
  hparams.learning_rate_constant = 1e-4
  hparams.clip_grad_norm = 0.5
  hparams.weight_decay = 0
  # If set, extends the LR warmup to all epochs except the final one.
  hparams.add_hparam("lr_decay_in_final_epoch", False)
  hparams.add_hparam("init_mean_factor", 0.1)
  hparams.add_hparam("init_logstd", 0.1)
  hparams.add_hparam("policy_layers", (100, 100))
  hparams.add_hparam("value_layers", (100, 100))
  hparams.add_hparam("clipping_coef", 0.2)
  hparams.add_hparam("gae_gamma", 0.99)
  hparams.add_hparam("gae_lambda", 0.95)
  hparams.add_hparam("entropy_loss_coef", 0.01)
  hparams.add_hparam("value_loss_coef", 1)
  hparams.add_hparam("optimization_epochs", 15)
  hparams.add_hparam("epoch_length", 200)
  hparams.add_hparam("epochs_num", 2000)
  hparams.add_hparam("eval_every_epochs", 10)
  hparams.add_hparam("save_models_every_epochs", 30)
  hparams.add_hparam("optimization_batch_size", 50)
  hparams.add_hparam("intrinsic_reward_scale", 0.)
  hparams.add_hparam("logits_clip", 0.0)
  hparams.add_hparam("dropout_ppo", 0.1)
  hparams.add_hparam("effective_num_agents", None)
  hparams.add_hparam("use_epochs", True)
  # TODO(afrozm): Clean this up, this is used in PPO learner to get modalities.
  hparams.add_hparam("policy_problem_name", "dummy_policy_problem")
  return hparams


@registry.register_hparams
def basic_policy_parameters():
  wrappers = None
  return hparam.HParams(wrappers=wrappers)


@registry.register_hparams
def ppo_discrete_action_base():
  hparams = ppo_base_v1()
  hparams.add_hparam("policy_network", "feed_forward_categorical_policy")
  return hparams


@registry.register_hparams
def discrete_random_action_base():
  hparams = common_hparams.basic_params1()
  hparams.add_hparam("policy_network", "random_policy")
  return hparams


@registry.register_hparams
def ppo_atari_base():
  """Pong base parameters."""
  hparams = ppo_discrete_action_base()
  hparams.learning_rate_constant = 1e-4
  hparams.epoch_length = 200
  hparams.gae_gamma = 0.985
  hparams.gae_lambda = 0.985
  hparams.entropy_loss_coef = 0.003
  hparams.value_loss_coef = 1
  hparams.optimization_epochs = 3
  hparams.epochs_num = 1000
  hparams.policy_network = "feed_forward_cnn_small_categorical_policy"
  hparams.clipping_coef = 0.2
  hparams.optimization_batch_size = 20
  hparams.clip_grad_norm = 0.5
  return hparams


@registry.register_hparams
def ppo_original_params():
  """Parameters based on the original PPO paper."""
  hparams = ppo_atari_base()
  hparams.learning_rate_constant = 2.5e-4
  hparams.gae_gamma = 0.99
  hparams.gae_lambda = 0.95
  hparams.clipping_coef = 0.1
  hparams.value_loss_coef = 1
  hparams.entropy_loss_coef = 0.01
  hparams.eval_every_epochs = 200
  hparams.dropout_ppo = 0.1
  # The parameters below are modified to accommodate short epoch_length (which
  # is needed for model based rollouts).
  hparams.epoch_length = 50
  hparams.optimization_batch_size = 20
  return hparams


@registry.register_hparams
def ppo_dist_params():
  """Parameters based on the original paper modified for distributional RL."""
  hparams = ppo_original_params()
  hparams.learning_rate_constant = 1e-3
  return hparams


@registry.register_hparams
def ppo_original_tiny():
  """Parameters based on the original PPO paper, tiny version."""
  hparams = ppo_original_params()
  hparams.epoch_length = 5
  hparams.optimization_batch_size = 1
  return hparams


@registry.register_hparams
def ppo_ttt_params():
  """Parameters based on the original PPO paper."""
  hparams = ppo_original_tiny()
  hparams.policy_network = "feed_forward_categorical_policy"
  hparams.policy_problem_name = "dummy_policy_problem_ttt"
  return hparams


@registry.register_hparams
def ppo_original_params_gamma95():
  """Parameters based on the original PPO paper, changed gamma."""
  hparams = ppo_original_params()
  hparams.gae_gamma = 0.95
  return hparams


@registry.register_hparams
def ppo_original_params_gamma90():
  """Parameters based on the original PPO paper, changed gamma."""
  hparams = ppo_original_params()
  hparams.gae_gamma = 0.90
  return hparams


@registry.register_hparams
def ppo_original_world_model():
  """Atari parameters with world model as policy."""
  hparams = ppo_original_params()
  hparams.policy_network = "next_frame_basic_deterministic"
  hparams_keys = hparams.values().keys()
  video_hparams = basic_deterministic_params.next_frame_basic_deterministic()
  for (name, value) in six.iteritems(video_hparams.values()):
    if name in hparams_keys:
      hparams.set_hparam(name, value)
    else:
      hparams.add_hparam(name, value)
  # Mostly to avoid decaying WM params when training the policy.
  hparams.weight_decay = 0
  return hparams


@registry.register_hparams
def ppo_tiny_world_model():
  """Atari parameters with world model as policy."""
  hparams = ppo_original_params()
  hparams.policy_network = "next_frame_basic_deterministic"
  hparams_keys = hparams.values().keys()
  video_hparams = basic_deterministic_params.next_frame_tiny()
  for (name, value) in six.iteritems(video_hparams.values()):
    if name in hparams_keys:
      hparams.set_hparam(name, value)
    else:
      hparams.add_hparam(name, value)
  hparams.weight_decay = 0
  return hparams


@registry.register_hparams
def ppo_original_world_model_stochastic_discrete():
  """Atari parameters with stochastic discrete world model as policy."""
  hparams = ppo_original_params()
  hparams.policy_network = "next_frame_basic_stochastic_discrete"
  hparams_keys = hparams.values().keys()
  video_hparams = basic_stochastic.next_frame_basic_stochastic_discrete()
  for (name, value) in six.iteritems(video_hparams.values()):
    if name in hparams_keys:
      hparams.set_hparam(name, value)
    else:
      hparams.add_hparam(name, value)
  # To avoid OOM. Probably way to small.
  hparams.optimization_batch_size = 1
  hparams.weight_decay = 0
  return hparams


def make_real_env_fn(env):
  """Creates a function returning a given real env, in or out of graph.

  Args:
    env: Environment to return from the function.

  Returns:
    Function in_graph -> env.
  """
  return lambda in_graph: PyFuncBatchEnv(env) if in_graph else env


def make_simulated_env_fn(**env_kwargs):
  """Returns a function creating a simulated env, in or out of graph.

  Args:
    **env_kwargs: kwargs to pass to the simulated env constructor.

  Returns:
    Function in_graph -> env.
  """
  def env_fn(in_graph):
    class_ = SimulatedBatchEnv if in_graph else SimulatedBatchGymEnv
    return class_(**env_kwargs)
  return env_fn


# TODO(koz4k): Move this and the one below to rl_utils.
def make_simulated_env_kwargs(real_env, hparams, **extra_kwargs):
  """Extracts simulated env kwargs from real_env and loop hparams."""
  objs_and_attrs = [
      (real_env, [
          "reward_range", "observation_space", "action_space", "frame_height",
          "frame_width"
      ]),
      (hparams, ["frame_stack_size", "intrinsic_reward_scale"])
  ]
  kwargs = {
      attr: getattr(obj, attr)  # pylint: disable=g-complex-comprehension
      for (obj, attrs) in objs_and_attrs for attr in attrs
  }
  kwargs["model_name"] = hparams.generative_model
  kwargs["model_hparams"] = trainer_lib.create_hparams(
      hparams.generative_model_params
  )
  if hparams.wm_policy_param_sharing:
    kwargs["model_hparams"].optimizer_zero_grads = True
  kwargs.update(extra_kwargs)
  return kwargs


def make_simulated_env_fn_from_hparams(real_env, hparams, **extra_kwargs):
  """Creates a simulated env_fn."""
  return make_simulated_env_fn(
      **make_simulated_env_kwargs(real_env, hparams, **extra_kwargs)
  )


def get_policy(observations, hparams, action_space,
               distributional_size=1, epoch=-1):
  """Get a policy network.

  Args:
    observations: observations
    hparams: parameters
    action_space: action space
    distributional_size: optional number of buckets for distributional RL
    epoch: optional epoch number

  Returns:
    Tuple (action logits, value).
  """
  if not isinstance(action_space, gym.spaces.Discrete):
    raise ValueError("Expecting discrete action space.")

  obs_shape = common_layers.shape_list(observations)
  (frame_height, frame_width) = obs_shape[2:4]

  # TODO(afrozm): We have these dummy problems mainly for hparams, so cleanup
  # when possible and do this properly.
  if hparams.policy_problem_name == "dummy_policy_problem_ttt":
    tf.logging.info("Using DummyPolicyProblemTTT for the policy.")
    policy_problem = tic_tac_toe_env.DummyPolicyProblemTTT()
  else:
    tf.logging.info("Using DummyPolicyProblem for the policy.")
    policy_problem = DummyPolicyProblem(action_space, frame_height, frame_width)

  trainer_lib.add_problem_hparams(hparams, policy_problem)
  hparams.force_full_predict = True
  model = registry.model(hparams.policy_network)(
      hparams, tf.estimator.ModeKeys.TRAIN
  )
  try:
    num_target_frames = hparams.video_num_target_frames
  except AttributeError:
    num_target_frames = 1
  target_value_shape_suffix = [num_target_frames]
  if distributional_size > 1:
    target_value_shape_suffix = [num_target_frames, distributional_size]
  features = {
      "inputs": observations,
      "epoch": tf.constant(epoch + 1),
      "input_action": tf.zeros(obs_shape[:2] + [1], dtype=tf.int32),
      "input_reward": tf.zeros(obs_shape[:2] + [1], dtype=tf.int32),
      "targets": tf.zeros(obs_shape[:1] + [num_target_frames] + obs_shape[2:]),
      "target_action": tf.zeros(
          obs_shape[:1] + [num_target_frames, 1], dtype=tf.int32),
      "target_reward": tf.zeros(
          obs_shape[:1] + [num_target_frames, 1], dtype=tf.int32),
      "target_policy": tf.zeros(
          obs_shape[:1] + [num_target_frames] + [action_space.n]),
      "target_value": tf.zeros(
          obs_shape[:1] + target_value_shape_suffix)
  }
  model.distributional_value_size = max(distributional_size, 1)
  model.use_epochs = hparams.use_epochs
  with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    t2t_model.create_dummy_vars()
    (targets, _) = model(features)
  target_values = targets["target_value"][:, 0]
  if distributional_size > 1:
    target_values = targets["target_value"][:, :]
  return (targets["target_policy"][:, 0, :], target_values)


@registry.register_hparams
def ppo_pong_ae_base():
  """Pong autoencoder base parameters."""
  hparams = ppo_original_params()
  hparams.learning_rate_constant = 1e-4
  hparams.network = "dense_bitwise_categorical_policy"
  return hparams


@registry.register_hparams
def dqn_atari_base():
  # These params are based on agents/dqn/configs/dqn.gin
  # with some modifications taking into account our code
  return hparam.HParams(
      agent_gamma=0.99,
      agent_update_horizon=1,
      agent_min_replay_history=20000,  # agent steps
      agent_update_period=4,
      agent_target_update_period=8000,  # agent steps
      agent_epsilon_train=0.01,
      agent_epsilon_eval=0.001,
      agent_epsilon_decay_period=250000,  # agent steps
      agent_generates_trainable_dones=True,
      agent_type="VanillaDQN",  # one of ["Rainbow", "VanillaDQN"]

      optimizer_class="RMSProp",
      optimizer_learning_rate=0.00025,
      optimizer_decay=0.95,
      optimizer_momentum=0.0,
      optimizer_epsilon=0.00001,
      optimizer_centered=True,

      # TODO(kozak): change names maybe replay_buffer -> agent?
      # Also batch_size is now buffer_batch_size in _DQNAgent.
      replay_buffer_replay_capacity=1000000,
      replay_buffer_buffer_batch_size=32,

      time_limit=27000,
      save_every_steps=50000,
      num_frames=int(20 * 1e6),

      # TODO(konradczechowski) this is not used in trainer_model_free, clean
      # this up after evaluation refactor
      eval_episodes_num=3,
  )


@registry.register_hparams
def dqn_original_params():
  """dqn_original_params."""
  hparams = dqn_atari_base()
  hparams.set_hparam("num_frames", int(1e6))
  return hparams


@registry.register_hparams
def dqn_guess1_params():
  """Guess 1 for DQN params."""
  hparams = dqn_atari_base()
  hparams.set_hparam("num_frames", int(1e6))
  hparams.set_hparam("agent_update_period", 1)
  hparams.set_hparam("agent_target_update_period", 400)
  # Small replay buffer size was set for mistake, but it seems to work
  hparams.set_hparam("replay_buffer_replay_capacity", 10000)
  return hparams


@registry.register_hparams
def dqn_guess1_params_eval():
  """Params for dqn_guess1 evaluation (with evaluator.py)."""
  hparams = dqn_guess1_params()
  hparams.set_hparam("eval_episodes_num", 64)
  return hparams


@registry.register_hparams
def dqn_guess1_rainbow_params():
  """Guess 1 for DQN params."""
  hparams = dqn_guess1_params()
  hparams.set_hparam("agent_type", "Rainbow")
  return hparams


@registry.register_hparams
def dqn_rainbow_params():
  """Rainbow params."""
  hparams = dqn_guess1_params()
  hparams.set_hparam("agent_type", "Rainbow")
  hparams.set_hparam("replay_buffer_replay_capacity", int(2e6) + int(1e5))
  return hparams


@registry.register_hparams
def dqn_2m_replay_buffer_params():
  """Guess 1 for DQN params, 2 milions transitions in replay buffer."""
  hparams = dqn_guess1_params()
  hparams.set_hparam("replay_buffer_replay_capacity", int(2e6) + int(1e5))
  return hparams


@registry.register_hparams
def dqn_10m_replay_buffer_params():
  """Guess 1 for DQN params, 10 milions transitions in replay buffer."""
  hparams = dqn_guess1_params()
  hparams.set_hparam("replay_buffer_replay_capacity", int(10e6))
  return hparams


def rlmf_tiny_overrides():
  """Parameters to override for tiny setting excluding agent-related hparams."""
  return dict(
      max_num_noops=1,
      eval_max_num_noops=1,
      rl_env_max_episode_steps=7,
      eval_rl_env_max_episode_steps=7,
      eval_sampling_temps=[0.0, 1.0],
  )


@registry.register_hparams
def rlmf_original():
  return hparam.HParams(
      game="pong",
      sticky_actions=False,
      base_algo="ppo",
      base_algo_params="ppo_original_params",
      batch_size=16,
      eval_batch_size=2,
      frame_stack_size=4,
      eval_sampling_temps=[0.0, 0.2, 0.5, 0.8, 1.0, 2.0],
      max_num_noops=8,
      eval_max_num_noops=8,
      eval_rl_env_max_episode_steps=1000,
      resize_height_factor=2,
      resize_width_factor=2,
      distributional_size=1,  # In distributional RL, number of buckets.
      distributional_subscale=0.04,  # How to scale values to buckets.
      distributional_threshold=0.0,  # Optimism threshold for experiments.
      grayscale=0,
      rl_env_max_episode_steps=-1,
      # If set, use this as the gym env name, instead of changing game mode etc.
      rl_env_name="",
      # Controls whether we should derive observation space, do some
      # pre-processing etc. See T2TGymEnv._derive_observation_space.
      rl_should_derive_observation_space=True,
      aunused=0,  # unused param for multi-run settings.
  )


@registry.register_hparams
def rlmf_tictactoe():
  """Base set of hparams for model-free PPO."""
  hparams = rlmf_original()
  hparams.game = "tictactoe"
  hparams.rl_env_name = "T2TEnv-TicTacToeEnv-v0"
  # Since we don't have any no-op actions, otherwise we have to have an
  # attribute called `get_action_meanings`.
  hparams.eval_max_num_noops = 0
  hparams.max_num_noops = 0
  hparams.rl_should_derive_observation_space = False

  hparams.policy_network = "feed_forward_categorical_policy"
  hparams.base_algo_params = "ppo_ttt_params"

  # Number of last observations to feed to the agent
  hparams.frame_stack_size = 1
  return hparams


@registry.register_hparams
def rlmf_base():
  """Base set of hparams for model-free PPO."""
  hparams = rlmf_original()
  hparams.add_hparam("ppo_epochs_num", 3000)
  hparams.add_hparam("ppo_eval_every_epochs", 100)
  return hparams


@registry.register_ranged_hparams
def rlmf_5runs(rhp):
  rhp.set_discrete("aunused", list(range(5)))


@registry.register_ranged_hparams
def rlmf_5runs_atari(rhp):
  rhp.set_categorical("game", gym_env.ATARI_GAMES_WITH_HUMAN_SCORE_NICE)
  rhp.set_discrete("aunused", list(range(5)))


@registry.register_hparams
def rlmf_dist():
  """Distributional set of hparams for model-free PPO."""
  hparams = rlmf_original()
  hparams.distributional_size = 1024
  hparams.base_algo_params = "ppo_dist_params"
  return hparams


@registry.register_hparams
def rlmf_dist_threshold():
  """Distributional set of hparams for model-free PPO."""
  hparams = rlmf_dist()
  hparams.distributional_threshold = 0.5
  return hparams


@registry.register_hparams
def rlmf_tiny():
  """Tiny set of hparams for model-free PPO."""
  hparams = rlmf_original()
  hparams = hparams.override_from_dict(rlmf_tiny_overrides())
  hparams.batch_size = 2
  hparams.base_algo_params = "ppo_original_tiny"
  hparams.add_hparam("ppo_epochs_num", 3)
  hparams.add_hparam("ppo_epoch_length", 2)
  return hparams


@registry.register_hparams
def rlmf_dqn_tiny():
  """Tiny DQN params."""
  hparams = rlmf_original()
  hparams = hparams.override_from_dict(rlmf_tiny_overrides())
  hparams.batch_size = 1
  hparams.base_algo = "dqn"
  hparams.base_algo_params = "dqn_original_params"
  hparams.add_hparam("dqn_num_frames", 128)
  hparams.add_hparam("dqn_save_every_steps", 128)
  hparams.add_hparam("dqn_replay_buffer_replay_capacity", 100)
  hparams.add_hparam("dqn_agent_min_replay_history", 10)
  return hparams


@registry.register_hparams
def rlmf_eval():
  """Eval set of hparams for model-free PPO."""
  hparams = rlmf_original()
  hparams.batch_size = 16
  hparams.eval_batch_size = 32
  hparams.eval_episodes_num = 2
  hparams.eval_sampling_temps = [0.5, 0.0, 1.0]
  hparams.eval_rl_env_max_episode_steps = 40000
  hparams.add_hparam("ppo_epoch_length", 128)
  hparams.add_hparam("ppo_optimization_batch_size", 32)
  hparams.add_hparam("ppo_epochs_num", 10000)
  hparams.add_hparam("ppo_eval_every_epochs", 500)
  hparams.add_hparam("attempt", 0)
  hparams.add_hparam("moe_loss_coef", 0)
  return hparams


@registry.register_hparams
def rlmf_eval_dist():
  """Distributional set of hparams for model-free PPO."""
  hparams = rlmf_eval()
  hparams.distributional_size = 4096
  hparams.distributional_subscale = 0.08
  hparams.base_algo_params = "ppo_dist_params"
  return hparams


@registry.register_hparams
def rlmf_eval_dist_threshold():
  """Distributional set of hparams for model-free PPO."""
  hparams = rlmf_eval_dist()
  hparams.distributional_threshold = 0.5
  return hparams


class PolicyBase(t2t_model.T2TModel):

  def __init__(self, *args, **kwargs):
    super(PolicyBase, self).__init__(*args, **kwargs)
    self.distributional_value_size = 1
    self.use_epochs = False

  def loss(self, *args, **kwargs):
    return 0.0


# TODO(lukaszkaiser): move this class or clean up the whole file.
class DummyPolicyProblem(video_utils.VideoProblem):
  """Dummy Problem for running the policy."""

  def __init__(self, action_space, frame_height, frame_width):
    super(DummyPolicyProblem, self).__init__()
    self.action_space = action_space
    self._frame_height = frame_height
    self._frame_width = frame_width

  @property
  def frame_height(self):
    """Height of each frame."""
    return self._frame_height

  @property
  def frame_width(self):
    """Width of each frame."""
    return self._frame_width

  @property
  def num_actions(self):
    return self.action_space.n

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.modality = {
        "inputs": modalities.ModalityType.VIDEO,
        "input_action": modalities.ModalityType.SYMBOL_WEIGHTS_ALL,
        "input_reward": modalities.ModalityType.SYMBOL_WEIGHTS_ALL,
        "targets": modalities.ModalityType.VIDEO,
        "target_action": modalities.ModalityType.SYMBOL_WEIGHTS_ALL,
        "target_reward": modalities.ModalityType.SYMBOL_WEIGHTS_ALL,
        "target_policy": modalities.ModalityType.IDENTITY,
        "target_value": modalities.ModalityType.IDENTITY,
    }
    p.vocab_size = {
        "inputs": 256,
        "input_action": self.num_actions,
        "input_reward": 3,
        "targets": 256,
        "target_action": self.num_actions,
        "target_reward": 3,
        "target_policy": None,
        "target_value": None,
    }
    p.input_space_id = problem.SpaceID.IMAGE
    p.target_space_id = problem.SpaceID.IMAGE


NetworkOutput = collections.namedtuple(
    "NetworkOutput", "policy, value, action_postprocessing")


# TODO(koz4k): Translate it to T2TModel or remove.
def feed_forward_gaussian_fun(action_space, config, observations):
  """Feed-forward Gaussian."""
  if not isinstance(action_space, gym.spaces.box.Box):
    raise ValueError("Expecting continuous action space.")

  mean_weights_initializer = tf.initializers.variance_scaling(
      scale=config.init_mean_factor)
  logstd_initializer = tf.random_normal_initializer(config.init_logstd, 1e-10)

  flat_observations = tf.reshape(observations, [
      tf.shape(observations)[0], tf.shape(observations)[1],
      functools.reduce(operator.mul, observations.shape.as_list()[2:], 1)])

  with tf.variable_scope("network_parameters"):
    with tf.variable_scope("policy"):
      x = flat_observations
      for size in config.policy_layers:
        x = tf.layers.dense(x, size, activation=tf.nn.relu)
      mean = tf.layers.dense(
          x, action_space.shape[0], activation=tf.tanh,
          kernel_initializer=mean_weights_initializer)
      logstd = tf.get_variable(
          "logstd", mean.shape[2:], tf.float32, logstd_initializer)
      logstd = tf.tile(
          logstd[None, None],
          [tf.shape(mean)[0], tf.shape(mean)[1]] + [1] * (mean.shape.ndims - 2))
    with tf.variable_scope("value"):
      x = flat_observations
      for size in config.value_layers:
        x = tf.layers.dense(x, size, activation=tf.nn.relu)
      value = tf.layers.dense(x, 1)[..., 0]
  mean = tf.check_numerics(mean, "mean")
  logstd = tf.check_numerics(logstd, "logstd")
  value = tf.check_numerics(value, "value")

  policy = tfp.distributions.MultivariateNormalDiag(mean, tf.exp(logstd))

  return NetworkOutput(policy, value, lambda a: tf.clip_by_value(a, -2., 2))


def clip_logits(logits, config):
  logits_clip = getattr(config, "logits_clip", 0.)
  if logits_clip > 0:
    min_logit = tf.reduce_min(logits)
    return tf.minimum(logits - min_logit, logits_clip)
  else:
    return logits


@registry.register_model
class FeedForwardCategoricalPolicy(PolicyBase):
  """Feed-forward categorical."""

  def body(self, features):
    observations = features["inputs_raw"]
    observations = tf.cast(observations, tf.float32)
    flat_observations = tf.layers.flatten(observations)
    with tf.variable_scope("policy"):
      x = flat_observations
      for size in self.hparams.policy_layers:
        x = tf.layers.dense(x, size, activation=tf.nn.relu)
      logits = tf.layers.dense(x, self.hparams.problem.num_actions)
      logits = tf.expand_dims(logits, axis=1)
    with tf.variable_scope("value"):
      x = flat_observations
      for size in self.hparams.value_layers:
        x = tf.layers.dense(x, size, activation=tf.nn.relu)
      value = tf.layers.dense(x, 1)
    logits = clip_logits(logits, self.hparams)
    return {"target_policy": logits, "target_value": value}


@registry.register_model
class FeedForwardCnnSmallCategoricalPolicy(PolicyBase):
  """Small cnn network with categorical output."""

  def body(self, features):
    observations = features["inputs_raw"]
    # Axis 0    - Batch.
    # Axis 1    - Input Frames, 4 frames.
    # Axis 2, 3 - Height & Width.
    # Axis 4    - Channels RGB, 3 colours.
    x = tf.transpose(observations, [0, 2, 3, 1, 4])
    x_shape = common_layers.shape_list(x)
    x = tf.reshape(x, x_shape[:-2] + [-1])
    dropout = getattr(self.hparams, "dropout_ppo", 0.0)
    with tf.variable_scope("feed_forward_cnn_small"):
      x = tf.cast(x, tf.float32) / 255.0
      x = tf.layers.conv2d(x, 32, (5, 5), strides=(2, 2),
                           activation=tf.nn.relu, padding="same")
      x = tf.layers.conv2d(x, 32, (5, 5), strides=(2, 2),
                           activation=tf.nn.relu, padding="same")

      flat_x = tf.layers.flatten(x)
      if self.use_epochs:
        epoch = features["epoch"] + tf.zeros([x_shape[0]], dtype=tf.int32)
        # Randomly set epoch to 0 in some cases as that's the inference value.
        rand = tf.random.uniform([x_shape[0]])
        epoch = tf.where(rand < 0.1, tf.zeros_like(epoch), epoch)
        # Embed the epoch number.
        emb_epoch = common_layers.embedding(epoch, 32, 32)  # [batch, 32]
        flat_x = tf.concat([flat_x, emb_epoch], axis=1)
      flat_x = tf.layers.dropout(flat_x, rate=dropout)
      x = tf.layers.dense(flat_x, 128, activation=tf.nn.relu)

      logits = tf.layers.dense(
          x, self.hparams.problem.num_actions, name="dense2"
      )
      logits = clip_logits(logits, self.hparams)
      logits = tf.expand_dims(logits, axis=1)
      value = tf.layers.dense(x, self.distributional_value_size)
    return {"target_policy": logits, "target_value": value}


@registry.register_model
class FeedForwardCnnSmallCategoricalPolicyNew(PolicyBase):
  """Small cnn network with categorical output."""

  def body(self, features):
    observations = features["inputs"]
    x = tf.transpose(observations, [0, 2, 3, 1, 4])
    x_shape = common_layers.shape_list(x)
    x = tf.reshape(x, x_shape[:-2] + [-1])
    dropout = getattr(self.hparams, "dropout_ppo", 0.0)
    with tf.variable_scope("feed_forward_cnn_small"):
      x = tf.cast(x, tf.float32) / 255.0
      x = tf.nn.dropout(x, rate=dropout)
      x = tf.layers.conv2d(
          x, 32, (4, 4), strides=(2, 2), name="conv1",
          activation=common_layers.belu, padding="SAME")
      x = tf.nn.dropout(x, rate=dropout)
      x = tf.layers.conv2d(
          x, 64, (4, 4), strides=(2, 2), name="conv2",
          activation=common_layers.belu, padding="SAME")
      x = tf.nn.dropout(x, rate=dropout)
      x = tf.layers.conv2d(
          x, 128, (4, 4), strides=(2, 2), name="conv3",
          activation=common_layers.belu, padding="SAME")

      flat_x = tf.layers.flatten(x)
      flat_x = tf.nn.dropout(flat_x, rate=dropout)
      x = tf.layers.dense(flat_x, 128, activation=tf.nn.relu, name="dense1")

      logits = tf.layers.dense(
          x, self.hparams.problem.num_actions, name="dense2"
      )
      logits = tf.expand_dims(logits, axis=1)
      logits = clip_logits(logits, self.hparams)

      value = tf.layers.dense(x, 1, name="value")
    return {"target_policy": logits, "target_value": value}


@registry.register_model
class DenseBitwiseCategoricalPolicy(PolicyBase):
  """Dense network with bitwise input and categorical output."""

  def body(self, features):
    observations = features["inputs"]
    flat_x = tf.layers.flatten(observations)
    with tf.variable_scope("dense_bitwise"):
      flat_x = discretization.int_to_bit_embed(flat_x, 8, 32)

      x = tf.layers.dense(flat_x, 256, activation=tf.nn.relu)
      x = tf.layers.dense(flat_x, 128, activation=tf.nn.relu)

      logits = tf.layers.dense(x, self.hparams.problem.num_actions)

      value = tf.layers.dense(x, 1)[..., 0]

    return {"target_policy": logits, "target_value": value}


@registry.register_model
class RandomPolicy(PolicyBase):
  """Random policy with categorical output."""

  def body(self, features):
    observations = features["inputs"]
    obs_shape = observations.shape.as_list()
    # Just so Saver doesn't complain because of no variables.
    tf.get_variable("dummy_var", initializer=0.0)
    num_actions = self.hparams.problem.num_actions
    logits = tf.constant(
        1. / float(num_actions),
        shape=(obs_shape[:1] + [1, num_actions])
    )
    value = tf.zeros(obs_shape[:1] + [1])
    return {"target_policy": logits, "target_value": value}
