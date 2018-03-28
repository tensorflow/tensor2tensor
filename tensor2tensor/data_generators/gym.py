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

"""Data generators for Gym environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

# Dependency imports
from collections import deque

import gym
import os
from tensorflow.contrib.training import HParams

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.models.research import rl
from tensor2tensor.utils import registry
from moviepy.editor import *
from tensor2tensor.rl.envs.utils import batch_env_factory
from tensor2tensor.rl.envs.tf_atari_wrappers import MemoryWrapper, TimeLimitWrapper
from tensor2tensor.rl.envs.tf_atari_wrappers import MaxAndSkipWrapper
from tensor2tensor.rl.envs.tf_atari_wrappers import PongT2TGeneratorHackWrapper
from tensor2tensor.rl import collect

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("agent_policy_path", "", "File with model for pong")

@registry.register_problem
class GymDiscreteProblem(problem.Problem):
  """Gym environment with discrete actions and rewards."""

  def __init__(self, *args, **kwargs):
    super(GymDiscreteProblem, self).__init__(*args, **kwargs)
    self.num_channels = 3
    self.history_size = 2

    # defaults
    self.environment_spec = lambda: gym.make("PongNoFrameskip-v4")
    self.in_graph_wrappers = [(MaxAndSkipWrapper, {"skip": 4})]
    self.collect_hparams = rl.atari_base()
    self.num_steps = 1000
    self.movies = True
    self.simulated_environment = None

  def _setup(self):
    # TODO: remove PongT2TGeneratorHackWrapper by writing a modality

    in_graph_wrappers = [(PongT2TGeneratorHackWrapper, {"add_value": 2}),
                         (MemoryWrapper, {})] + self.in_graph_wrappers
    env_hparams = HParams(in_graph_wrappers=in_graph_wrappers,
                          simulated_environment=self.simulated_environment)

    generator_batch_env = \
      batch_env_factory(self.environment_spec, env_hparams, num_agents=1, xvfb=False)


    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
      policy_lambda = self.collect_hparams.network
      policy_factory = tf.make_template(
        "network",
        functools.partial(policy_lambda, self.environment_spec().action_space, self.collect_hparams),
        create_scope_now_=True,
        unique_name_="network")

    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
      sample_policy = lambda policy: policy.sample()

      self.collect_hparams.epoch_length = 10
      _, self.collect_trigger_op = collect.define_collect(
        policy_factory, generator_batch_env, self.collect_hparams,
        eval_phase=False, policy_to_actions_lambda=sample_policy, scope="define_collect")

    self.avilable_data_size_op = MemoryWrapper.singleton._speculum.size()
    self.data_get_op = MemoryWrapper.singleton._speculum.dequeue()
    self.history_buffer = deque(maxlen=self.history_size+1)

  def example_reading_spec(self, label_repr=None):
    data_fields = {

      "targets_encoded": tf.FixedLenFeature((), tf.string),
      "image/format": tf.FixedLenFeature((), tf.string),
      "action": tf.FixedLenFeature([1], tf.int64),
      "reward": tf.FixedLenFeature([1], tf.int64),
      # "done": tf.FixedLenFeature([1], tf.int64)
    }

    for x in range(self.history_size):
      data_fields["inputs_encoded_{}".format(x)] =  tf.FixedLenFeature((), tf.string)


    data_items_to_decoders = {
      "targets":
        tf.contrib.slim.tfexample_decoder.Image(
          image_key="targets_encoded",
          format_key="image/format",
          shape=[210, 160, 3],
          channels=3),

      #Just do a pass through
      "action":tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="action"),
      "reward":tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="reward"),
    }

    for x in range(self.history_size):
      data_items_to_decoders["inputs_{}".format(x)] =  tf.contrib.slim.tfexample_decoder.Image(
                image_key="inputs_encoded_{}".format(x),
                format_key="image/format",
                shape=[210, 160, 3],
                channels=3)


    return data_fields, data_items_to_decoders

  @property
  def num_actions(self):
    return 4

  @property
  def num_rewards(self):
    return 2

  @property
  def num_shards(self):
    return 10

  @property
  def num_dev_shards(self):
    return 1

  def get_action(self, observation=None):
    return self.env.action_space.sample()

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    # hard coded +1 after "symbol" refers to the fact
    # that 0 is a special symbol meaning padding
    # when symbols are e.g. 0, 1, 2, 3 we
    # shift them to 0, 1, 2, 3, 4
    #TODO: check if we do not need to change symbol:identity to symbol
    #TODO: verify if we handle actions correctly. Curretly not adding +1?
    p.input_modality = {"action": ("symbol:identity", self.num_actions)}

    for x in range(self.history_size):
      p.input_modality["inputs_{}".format(x)] = ("image", 256)

    p.target_modality = {"targets": ("image", 256),
                         "reward":  ("symbol", self.num_rewards+1),
                         # "done": ("symbol", 2+1)
                         }

    p.input_space_id = problem.SpaceID.IMAGE
    p.target_space_id = problem.SpaceID.IMAGE

  def restore_networks(self, sess):
    model_saver = tf.train.Saver(
      tf.global_variables(".*network_parameters.*"))
    if FLAGS.agent_policy_path:
      model_saver.restore(sess, FLAGS.agent_policy_path)

  def generator(self, data_dir, tmp_dir):
    self._setup()
    clip_files = []
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      self.restore_networks(sess)

      pieces_generated = 0
      while pieces_generated<self.num_steps:
        avilable_data_size = sess.run(self.avilable_data_size_op)
        if avilable_data_size>0:
          observ, reward, action, done = sess.run(self.data_get_op)
          self.history_buffer.append(observ)

          if self.movies==True:
            file_name = os.path.join(tmp_dir,'output_{}.png'.format(pieces_generated))
            clip_files.append(file_name)
            with open(file_name, 'wb') as f:
              f.write(observ)

          if len(self.history_buffer)==self.history_size+1:
            pieces_generated += 1
            ret_dict = {
              "targets_encoded": [observ],
              "image/format": ["png"],
              "action": [int(action)],
              # "done": [bool(done)],
              "reward": [int(reward)],
                }
            for i, v in enumerate(list(self.history_buffer)[:-1]):
              ret_dict["inputs_encoded_{}".format(i)] = [v]
            yield ret_dict
        else:
          sess.run(self.collect_trigger_op)
    if self.movies:
      clip = ImageSequenceClip(clip_files, fps=25)
      clip.write_videofile(os.path.join(data_dir, 'output_{}.mp4'.format(self.name)),
                           fps=25, codec='mpeg4')


  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    train_paths = self.training_filepaths(
        data_dir, self.num_shards, shuffled=False)
    dev_paths = self.dev_filepaths(
        data_dir, self.num_dev_shards, shuffled=False)
    all_paths = train_paths + dev_paths
    generator_utils.generate_files(
        self.generator(data_dir, tmp_dir), all_paths)
    generator_utils.shuffle_dataset(all_paths)


@registry.register_problem
class GymSimulatedDiscreteProblem(GymDiscreteProblem):
  """Simulated gym environment with discrete actions and rewards."""

  def __init__(self, *args, **kwargs):
    super(GymSimulatedDiscreteProblem, self).__init__(*args, **kwargs)
    #TODO: pull it outside
    self.in_graph_wrappers = [(TimeLimitWrapper, {"timelimit": 150}), (MaxAndSkipWrapper, {"skip": 4})]
    self.simulated_environment = True

  def restore_networks(self, sess):
    super(GymSimulatedDiscreteProblem, self).restore_networks(sess)

    #TODO: adjust regexp for different models
    env_model_loader = tf.train.Saver(tf.global_variables(".*basic_conv_gen.*"))
    sess = tf.get_default_session()

    ckpts = tf.train.get_checkpoint_state(FLAGS.output_dir)
    ckpt = ckpts.model_checkpoint_path
    env_model_loader.restore(sess, ckpt)

