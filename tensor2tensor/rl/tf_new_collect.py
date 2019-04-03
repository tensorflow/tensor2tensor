# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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

"""New collect."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import copy

from tensor2tensor.data_generators.gym_env import DummyWorldModelProblem
from tensor2tensor.layers import common_layers
from tensor2tensor.models.research.rl import get_policy
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib

import tensorflow as tf
import tensorflow_probability as tfp

from gym.spaces import Discrete


TensorSpec = namedtuple("TensorSpec", ("shape", "dtype"))
# Defining those as namedtuples may seem superfluous but it has documentational
# value, adds type-safety and enforces ordering which we wouldn't have using
# normal tuples or OrderedDicts.
EnvData = namedtuple(
    "EnvData", ("hidden_state", "observation", "action")
)
PPOData = namedtuple(
    "PPOData", ("observation", "reward", "done", "action", "pdf", "value")
)


def init_tensor_structure(spec):
  # spec is either TensorSpec or a tuple of specs.
  try:
    return tf.zeros(spec.shape, spec.dtype)
  except AttributeError:
    return tuple(map(init_tensor_structure, spec))


class NewInGraphBatchEnv(object):

  def __init__(self, batch_size):
    self.batch_size = batch_size

  @property
  def tensor_specs(self):
    raise NotImplementedError()

  @property
  def empty_hidden_state(self):
    raise NotImplementedError()

  def step(self, hidden_state, action):
    raise NotImplementedError()


class NewSimulatedBatchEnv(NewInGraphBatchEnv):

  def __init__(self, batch_size, model_name, model_hparams):
    super(NewSimulatedBatchEnv, self).__init__(batch_size)
    model_hparams = copy.copy(model_hparams)
    problem = DummyWorldModelProblem(
        action_space=Discrete(2), reward_range=(-1, 1),
        frame_height=210, frame_width=160
    )
    trainer_lib.add_problem_hparams(model_hparams, problem)
    model_hparams.force_full_predict = True
    self._model = registry.model(model_name)(
        model_hparams, tf.estimator.ModeKeys.PREDICT
    )

  @property
  def tensor_specs(self):
    return EnvData(
        hidden_state=TensorSpec(
            shape=(self.batch_size, 4, 210, 160, 3), dtype=tf.int32
        ),
        observation=TensorSpec(
            shape=(self.batch_size, 210, 160, 3), dtype=tf.int32
        ),
        action=TensorSpec(
            shape=(self.batch_size,), dtype=tf.int32
        ),
    )

  def step(self, hidden_state, action):
    history = hidden_state

    action = tf.stack([action] * 4, axis=1)
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      # We only need 1 target frame here, set it.
      hparams_target_frames = self._model.hparams.video_num_target_frames
      self._model.hparams.video_num_target_frames = 1

      if False:  # Change to True to run a stupid model (single conv).
        history_squeezed = tf.math.reduce_mean(history, axis=1)
        model_output = tf.layers.conv2d(
            tf.cast(history_squeezed, tf.float32),
            filters=3, kernel_size=(3, 3), padding='same'
        )
        model_output = {
            "targets": tf.expand_dims(model_output, axis=1),
            "target_reward": tf.ones((self.batch_size,)),
        }
      else:
        model_output = self._model.infer({
            "inputs": history,
            "input_action": action,
            "reset_internal_states": 0.0  # TODO: How?
        })

    self._model.hparams.video_num_target_frames = hparams_target_frames
    ob_unsqueezed = tf.cast(model_output["targets"], tf.int32)
    new_history = tf.concat([history[:, 1:, ...], ob_unsqueezed], axis=1)
    ob = tf.squeeze(ob_unsqueezed, axis=1)
    reward = tf.reshape(
        tf.cast(model_output["target_reward"], tf.float32),
        shape=(self.batch_size,)
    ) - 1

    return ((new_history,), ob, reward)

  def reset(self):
    specs = self.tensor_specs
    return tuple(map(
        init_tensor_structure,
        (specs.hidden_state, specs.observation)
    ))


class NewStackWrapper(NewInGraphBatchEnv):
  #TODO: it might be more natural to stack on the last dim

  def __init__(self, batch_env, history=4):
    super(NewStackWrapper, self).__init__(batch_env.batch_size)
    self._env = batch_env
    self.history = history

  @property
  def tensor_specs(self):
    specs = self._env.tensor_specs
    ob_spec = specs.observation
    stacked_ob_spec = ob_spec._replace(
        shape=(self.batch_size, self.history) + ob_spec.shape[1:]
    )
    return specs._replace(
        hidden_state=(stacked_ob_spec, specs.hidden_state),
        observation=stacked_ob_spec,
    )

  def step(self, hidden_state, action):
    (stack_hidden_state, env_hidden_state) = hidden_state
    (new_env_hidden_state, env_ob, env_reward) = self._env.step(
        env_hidden_state, action
    )
    new_stack_hidden_state = tf.concat(
        [stack_hidden_state[:, 1:, ...], tf.expand_dims(env_ob, axis=1)], axis=1
    )

    return (
        (new_stack_hidden_state, new_env_hidden_state),
        new_stack_hidden_state,
        env_reward
    )

  def reset(self):
    specs = self.tensor_specs
    return tuple(map(
        init_tensor_structure,
        (specs.hidden_state, specs.observation)
    ))


def new_define_collect(batch_env, hparams, action_space):
  batch_size = batch_env.batch_size
  env_tensor_specs = batch_env.tensor_specs
  ppo_tensor_specs = PPOData(
      observation=env_tensor_specs.observation,
      reward=TensorSpec(shape=(batch_size,), dtype=tf.float32),
      done=TensorSpec(shape=(batch_size,), dtype=tf.bool),
      action=env_tensor_specs.action,
      pdf=TensorSpec(shape=(batch_size,), dtype=tf.float32),
      value=TensorSpec(shape=(batch_size,), dtype=tf.float32),
  )

  # These are only for typing, values will be discarded
  initial_ppo_batch = PPOData(*init_tensor_structure(ppo_tensor_specs))

  (hidden_state, observation) = batch_env.reset()
  # TODO: Abstract this out.
  hidden_state = [
      tf.reshape(element, (-1,)) for element in hidden_state
  ]
  observation = tf.reshape(observation, (-1,))
  initial_running_state = (hidden_state, observation)
  initial_ppo_batch = initial_ppo_batch._replace(observation=observation)

  initial_batch = initial_running_state + (initial_ppo_batch,)

  def execution_wrapper(hidden_state, observation):
    # TODO: Abstract this out.
    hidden_state = [
        tf.reshape(element, spec.shape)
        for (element, spec) in zip(
            hidden_state, env_tensor_specs.hidden_state
        )
    ]
    observation = tf.reshape(observation, env_tensor_specs.observation.shape)

    (logits, value) = get_policy(observation, hparams, action_space)
    action = common_layers.sample_with_temperature(logits, 1)
    action = tf.cast(action, tf.int32)
    pdf = tfp.distributions.Categorical(logits=logits).prob(action)

    hidden_state, new_observation, reward = batch_env.step(
        hidden_state, action
    )

    # TODO: This too.
    hidden_state = [tf.reshape(x, (-1,)) for x in hidden_state]
    (new_observation, observation) = (
        tf.reshape(x, (-1,)) for x in (new_observation, observation)
    )
    done = tf.zeros((batch_size,), dtype=tf.bool)

    return (
        hidden_state, new_observation,
        PPOData(
            observation=observation,
            reward=reward,
            done=done,
            action=action,
            pdf=pdf,
            value=value
        )
    )

  # TODO: Replace with while and filling an array manually. Otherwise we have to
  # reshape the accumulator in each iteration so it can fit in an output array
  # which we discard anyway (we only keep ret[3]). Can gain up to 12%
  # performance by doing so.
  (_, _, ppo_data) = tf.scan(
      lambda running_state, _: execution_wrapper(*running_state[:2]),
      tf.range(hparams.epoch_length), initial_batch
  )

  return ppo_data
