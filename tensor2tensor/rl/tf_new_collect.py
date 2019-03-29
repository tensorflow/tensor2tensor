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

import copy

from tensor2tensor.data_generators.gym_env import DummyWorldModelProblem
from tensor2tensor.layers import common_layers
from tensor2tensor.models.research.rl import get_policy
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib

import tensorflow as tf
import tensorflow_probability as tfp

from gym.spaces import Discrete


class NewInGraphBatchEnv(object):

  def __init__(self, batch_size):
    self.batch_size = batch_size

  @property
  def meta_data(self):
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
  def meta_data(self):
    # TODO: namedtuples
    return (
        [([self.batch_size, 4, 210, 160, 3], tf.int32, "hidden_state")],
        ([self.batch_size, 210, 160, 3], tf.int32, "observation"),
        ([self.batch_size], tf.int32, "action"),
    )

  def step(self, hidden_state, action):
    (history,) = hidden_state

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
    done = tf.constant([False] * self.batch_size)

    return (new_history,), ob, reward, done

  def reset(self, hidden_state, observation, done):
    hidden_state_unpacked = hidden_state[0]
    # TODO: Always reset on first step.
    #new_hidden_state, observation, _ = tf.scan(
    #    lambda _, x: tf.cond(
    #        x[2],
    #        lambda: self._reset_one_env() + (tf.constant(False),),
    #        lambda: x
    #    ),
    #    (hidden_state_unpacked, observation, done),
    #    initializer=(hidden_state_unpacked[0], observation[0], done[0])
    #)
    new_hidden_state = hidden_state_unpacked
    return (new_hidden_state,), observation


  def _reset_one_env(self):
    hidden_state_single_env = tf.zeros(
        self.meta_data[0][0][0][1:],
        self.meta_data[0][0][1]
    )
    return hidden_state_single_env, hidden_state_single_env[-1, ...]


class NewStackWrapper(NewInGraphBatchEnv):
  #TODO: it might be more natural to stack on the last dim

  def __init__(self, batch_env, history=4):
    super(NewStackWrapper, self).__init__(batch_env.batch_size)
    self._env = batch_env
    self.history = history

  @property
  def meta_data(self):
    hs, ob, ac = self._env.meta_data
    ob_shape, ob_type, _ = ob
    stack_ob_shape = [self.batch_size, self.history] + ob_shape[1:]
    stack_ob_spec = (stack_ob_shape, ob_type, "observation")

    return hs + [stack_ob_spec], stack_ob_spec, ac

  def step(self, hidden_state, action):
    env_hidden_state, stack_hidden_state = hidden_state
    (new_env_hidden_state,), env_ob, env_reward, env_done = self._env.step(
        (env_hidden_state,), action
    )
    new_stack_hidden_state = tf.concat(
        [stack_hidden_state[:, 1:, ...], tf.expand_dims(env_ob, axis=1)], axis=1
    )

    return (
        (new_env_hidden_state, new_stack_hidden_state), new_stack_hidden_state,
        env_reward, env_done
    )

  def reset(self, hidden_state, _, done):
    env_hidden_state, stack_hidden_state = hidden_state
    env_observ = stack_hidden_state[:, -1, ...]
    (new_env_hidden_state,), new_env_observation = self._env.reset(
        (env_hidden_state,), env_observ, done
    )

    def extend(ob):
      _, (ob_shape, _, _), _ = self._env.meta_data
      multiples = (self.history,) + (1,) * (len(ob_shape) - 1)
      return tf.tile(tf.expand_dims(ob, axis=0), multiples)

    # TODO: Wrap this in a Python if.
    #new_stack_hidden_state, _, _ = tf.scan(
    #    lambda _, x: tf.cond(
    #        x[2],
    #        lambda: (extend(x[1]), x[1], tf.constant(False)),
    #        lambda: x
    #    ),
    #    (stack_hidden_state, new_env_observation, done),
    #    initializer=(stack_hidden_state[0], new_env_observation[0], done[0])
    #)
    new_stack_hidden_state = stack_hidden_state

    return (
        (new_env_hidden_state, new_stack_hidden_state), new_stack_hidden_state
    )


def new_define_collect(
    batch_env, hparams, action_space, force_beginning_resets
):
  batch_size = batch_env.batch_size
  hidden_state_types, observation_type, action_type = batch_env.meta_data
  done_type = ([batch_size], tf.bool, "done")

  ppo_data_metadata = [observation_type, ([batch_size], tf.float32, "reward"),
                       done_type, action_type,
                       ([batch_size], tf.float32, "pdf"),
                       ([batch_size], tf.float32, "value_function")]

  initial_state_metadata = hidden_state_types + [observation_type, done_type]

  # These are only for typing, values will be discarded
  initial_ppo_batch = tuple(
      tf.zeros(shape, dtype=type) for shape, type, _ in ppo_data_metadata
  )

  # Below we intialize with ones, to
  # set done=True. Other fields are just for typeing.
  if force_beginning_resets:
    initial_running_state = [
        tf.ones(shape, dtype=type) for shape, type, _ in initial_state_metadata
    ]
  else:
    initial_running_state = [
        tf.get_variable(  # pylint: disable=g-complex-comprehension
            "collect_initial_running_state_%s" % (name),
            shape=shape,
            dtype=dtype,
            initializer=tf.ones_initializer(),
            trainable=False
        )
        for (shape, dtype, name) in initial_state_metadata
    ]

  initial_running_state = tf.contrib.framework.nest.pack_sequence_as(
      ((1,) * len(hidden_state_types),) + (2, 3), initial_running_state)

  (hidden_state, observation, done) = initial_running_state
  hidden_state = [
      tf.reshape(element, (-1,)) for element in hidden_state
  ]
  observation = tf.reshape(observation, (-1,))
  initial_running_state = (hidden_state, observation, done)
  initial_ppo_batch = (observation, *initial_ppo_batch[1:])

  initial_batch = initial_running_state + (initial_ppo_batch,)

  def execution_wrapper(hidden_state, observation, done):
    # TODO: Abstract this out.
    hidden_state = [
        tf.reshape(element, meta_data[0])
        for (element, meta_data) in zip(hidden_state, hidden_state_types)
    ]
    observation = tf.reshape(observation, observation_type[0])

    hidden_state, observation = batch_env.reset(hidden_state, observation, done)
    (logits, value_function) = get_policy(observation, hparams, action_space)
    action = common_layers.sample_with_temperature(logits, 1)
    action = tf.cast(action, tf.int32)
    pdf = tfp.distributions.Categorical(logits=logits).prob(action)

    hidden_state, new_observation, reward, done = batch_env.step(
        hidden_state, action
    )

    # TODO: This too.
    hidden_state = [tf.reshape(x, (-1,)) for x in hidden_state]
    (new_observation, observation) = (
        tf.reshape(x, (-1,)) for x in (new_observation, observation)
    )

    return (
        hidden_state, new_observation, done,
        (observation, reward, done, action, pdf, value_function)
    )

  # TODO: Replace with while and filling an array manually. Otherwise we have to
  # reshape the accumulator in each iteration so it can fit in an output array
  # which we discard anyway (we only keep ret[3]). Can gain up to 12%
  # performance by doing so.
  ret = tf.scan(
      lambda running_state, _: execution_wrapper(*running_state[:3]),
      tf.range(hparams.epoch_length), initial_batch
  )

  return ret[3]
