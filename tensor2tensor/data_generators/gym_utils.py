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
"""Utilities for openai gym."""

from collections import deque
import gym

import numpy as np


# pylint: disable=method-hidden
class WarmupWrapper(gym.Wrapper):
  """Warmup wrapper."""

  def __init__(self, env, warm_up_examples=0, warmup_action=0):
    gym.Wrapper.__init__(self, env)
    self.warm_up_examples = warm_up_examples
    self.warm_up_action = warmup_action
    self.observation_space = gym.spaces.Box(
        low=0, high=255, shape=(210, 160, 3), dtype=np.uint8)

  def get_starting_data(self, num_frames):
    self.reset()
    starting_observations, starting_actions, starting_rewards = [], [], []
    for _ in range(num_frames):
      observation, rew, _, _ = self.env.step(self.warm_up_action)
      starting_observations.append(observation)
      starting_rewards.append(rew)
      starting_actions.append(self.warm_up_action)

    return starting_observations, starting_actions, starting_rewards

  def step(self, action):
    return self.env.step(action)

  def reset(self, **kwargs):
    del kwargs
    self.env.reset()
    observation = None
    for _ in range(self.warm_up_examples):
      observation, _, _, _ = self.env.step(self.warm_up_action)

    return observation


class PongWrapper(WarmupWrapper):
  """Pong Wrapper."""

  def __init__(self, env, warm_up_examples=0,
               action_space_reduction=False,
               reward_skip_steps=0,
               big_ball=False):
    super(PongWrapper, self).__init__(env, warm_up_examples=warm_up_examples)
    self.action_space_reduction = action_space_reduction
    if self.action_space_reduction:
      self.action_space = gym.spaces.Discrete(2)
    self.warm_up_examples = warm_up_examples
    self.observation_space = gym.spaces.Box(
        low=0, high=255, shape=(210, 160, 3), dtype=np.uint8)
    self.reward_skip_steps = reward_skip_steps
    self.big_ball = big_ball

  def step(self, action):
    if self.action_space_reduction:
      action = 2 if int(action) == 0 else 5
    ob, rew, done, info = self.env.step(action)
    ob = self.process_observation(ob)
    if rew != 0 and self.reward_skip_steps != 0:
      for _ in range(self.reward_skip_steps):
        self.env.step(0)
    return ob, rew, done, info

  def reset(self, **kwargs):
    observation = super(PongWrapper, self).reset(**kwargs)
    observation = self.process_observation(observation)
    return observation

  def process_observation(self, obs):
    if self.big_ball:
      pos = PongWrapper.find_ball(obs)
      if pos is not None:
        x, y = pos
        obs[x-5:x+5, y-5:y+5, :] = 255

    return obs

  @staticmethod
  def find_ball(obs, default=None):
    ball_area = obs[37:193, :, 0]
    res = np.argwhere(ball_area == 236)
    if not res:
      return default
    else:
      x, y = res[0]
      x += 37
      return x, y


def wrapped_pong_factory(warm_up_examples=0, action_space_reduction=False,
                         reward_skip_steps=0, big_ball=False):
  """Wrapped pong games."""
  env = gym.make("PongDeterministic-v4")
  env = env.env  # Remove time_limit wrapper.
  env = PongWrapper(env, warm_up_examples=warm_up_examples,
                    action_space_reduction=action_space_reduction,
                    reward_skip_steps=reward_skip_steps,
                    big_ball=big_ball)
  return env


gym.envs.register(id="T2TPongWarmUp20RewSkip200Steps-v1",
                  entry_point=lambda: wrapped_pong_factory(  # pylint: disable=g-long-lambda
                      warm_up_examples=20, reward_skip_steps=15),
                  max_episode_steps=200)


gym.envs.register(id="T2TPongWarmUp20RewSkip2000Steps-v1",
                  entry_point=lambda: wrapped_pong_factory(  # pylint: disable=g-long-lambda
                      warm_up_examples=20, reward_skip_steps=15),
                  max_episode_steps=2000)


class BreakoutWrapper(WarmupWrapper):
  """Breakout Wrapper."""

  FIRE_ACTION = 1

  def __init__(self, env, warm_up_examples=0,
               ball_down_skip=0,
               big_ball=False,
               include_direction_info=False,
               reward_clipping=True):
    super(BreakoutWrapper, self).__init__(
        env, warm_up_examples=warm_up_examples,
        warmup_action=BreakoutWrapper.FIRE_ACTION)
    self.warm_up_examples = warm_up_examples
    self.observation_space = gym.spaces.Box(low=0, high=255,
                                            shape=(210, 160, 3),
                                            dtype=np.uint8)
    self.ball_down_skip = ball_down_skip
    self.big_ball = big_ball
    self.reward_clipping = reward_clipping
    self.include_direction_info = include_direction_info
    self.direction_info = deque([], maxlen=2)
    self.points_gained = False
    msg = ("ball_down_skip should be bigger equal 9 for "
           "include_direction_info to work correctly")
    assert not self.include_direction_info or ball_down_skip >= 9, msg

  def step(self, action):
    ob, rew, done, info = self.env.step(action)

    if BreakoutWrapper.find_ball(ob) is None and self.ball_down_skip != 0:
      for _ in range(self.ball_down_skip):
        # We assume that nothing interesting happens during ball_down_skip
        # and discard all information.
        # We fire all the time to start new game
        ob, _, _, _ = self.env.step(BreakoutWrapper.FIRE_ACTION)
        self.direction_info.append(BreakoutWrapper.find_ball(ob))

    ob = self.process_observation(ob)

    self.points_gained = self.points_gained or rew > 0

    if self.reward_clipping:
      rew = np.sign(rew)

    return ob, rew, done, info

  def reset(self, **kwargs):
    observation = super(BreakoutWrapper, self).reset(**kwargs)
    self.env.step(BreakoutWrapper.FIRE_ACTION)
    self.direction_info = deque([], maxlen=2)
    observation = self.process_observation(observation)
    return observation

  @staticmethod
  def find_ball(ob, default=None):
    off_x = 63
    clipped_ob = ob[off_x:-21, :, 0]
    pos = np.argwhere(clipped_ob == 200)

    if not pos.size:
      return default

    x = off_x + pos[0][0]
    y = 0 + pos[0][1]
    return x, y

  def process_observation(self, obs):
    if self.big_ball:
      pos = BreakoutWrapper.find_ball(obs)
      if pos is not None:
        x, y = pos
        obs[x-5:x+5, y-5:y+5, :] = 255

    if self.include_direction_info:
      for point in list(self.direction_info):
        if point is not None:
          x, y = point
          obs[x-2:x+2, y-2:y+2, 1] = 255

    return obs


def wrapped_breakout_factory(warm_up_examples=0,
                             ball_down_skip=0,
                             big_ball=False,
                             include_direction_info=False,
                             reward_clipping=True):
  """Wrapped breakout games."""
  env = gym.make("BreakoutDeterministic-v4")
  env = env.env  # Remove time_limit wrapper.
  env = BreakoutWrapper(env, warm_up_examples=warm_up_examples,
                        ball_down_skip=ball_down_skip,
                        big_ball=big_ball,
                        include_direction_info=include_direction_info,
                        reward_clipping=reward_clipping)
  return env


gym.envs.register(id="T2TBreakoutWarmUp20RewSkip500Steps-v1",
                  entry_point=lambda: wrapped_breakout_factory(  # pylint: disable=g-long-lambda
                      warm_up_examples=1,
                      ball_down_skip=9,
                      big_ball=False,
                      include_direction_info=True,
                      reward_clipping=True
                  ),
                  max_episode_steps=500)


class FreewayWrapper(WarmupWrapper):
  """Wrapper for Freeway."""

  def __init__(self, env,
               warm_up_examples=0,
               reward_clipping=True,
               easy_freeway=False):
    super(FreewayWrapper, self).__init__(env, warm_up_examples)
    self.easy_freeway = easy_freeway
    self.half_way_reward = 1.0

    # this is probably not needed, just in case
    self.reward_clipping = reward_clipping

  def chicken_height(self, image):
    raise NotImplementedError()

  def step(self, action):
    ob, rew, done, info = self.env.step(action)

    if self.easy_freeway:
      if rew > 0:
        self.half_way_reward = 1
      chicken_height = self.chicken_height(ob)
      if chicken_height < 105:
        rew += self.half_way_reward
        self.half_way_reward = 0

    if self.reward_clipping:
      rew = np.sign(rew)

    return ob, rew, done, info

  def reset(self, **kwargs):
    self.half_way_reward = 1.0
    observation = super(FreewayWrapper, self).reset(**kwargs)
    return observation


def wrapped_freeway_factory(warm_up_examples=0,
                            reward_clipping=True,
                            easy_freeway=False):
  """Wrapped freeway games."""
  env = gym.make("FreewayDeterministic-v4")
  env = env.env  # Remove time_limit wrapper.
  env = FreewayWrapper(env, warm_up_examples=warm_up_examples,
                       reward_clipping=reward_clipping,
                       easy_freeway=easy_freeway)

  return env

gym.envs.register(id="T2TFreewayWarmUp20RewSkip500Steps-v1",
                  entry_point=lambda: wrapped_freeway_factory(  # pylint: disable=g-long-lambda
                      warm_up_examples=1,
                      reward_clipping=True,
                      easy_freeway=False
                  ),
                  max_episode_steps=500)
