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

"""Various wrappers copied for Gym Baselines."""

from collections import deque
import gym
import numpy as np


# Adapted from the link below.
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py


class WarpFrame(gym.ObservationWrapper):
  """Wrap a frame."""

  def __init__(self, env):
    """Warp frames to 84x84 as done in the Nature paper and later work."""
    gym.ObservationWrapper.__init__(self, env)
    self.width = 84
    self.height = 84
    self.observation_space = gym.spaces.Box(
        low=0, high=255,
        shape=(self.height, self.width, 1), dtype=np.uint8)

  def observation(self, frame):
    import cv2  # pylint: disable=g-import-not-at-top
    cv2.ocl.setUseOpenCL(False)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (self.width, self.height),
                       interpolation=cv2.INTER_AREA)
    return frame[:, :, None]


class LazyFrames(object):
  """Lazy frame storage."""

  def __init__(self, frames):
    """Lazy frame storage.

      This object ensures that common frames between the observations
      are only stored once. It exists purely to optimize memory usage
      which can be huge for DQN's 1M frames replay buffers.
      This object should only be converted to numpy array before being passed
      to the model.

    Args:
      frames: the frames.
    """
    self._frames = frames

  def __array__(self, dtype=None):
    out = np.concatenate(self._frames, axis=2)
    if dtype is not None:
      out = out.astype(dtype)
    return out


class FrameStack(gym.Wrapper):
  """Stack frames."""

  def __init__(self, env, k):
    """Stack k last frames. Returns lazy array, memory efficient."""
    gym.Wrapper.__init__(self, env)
    self.k = k
    self.frames = deque([], maxlen=k)
    shp = env.observation_space.shape
    self.observation_space = gym.spaces.Box(
        low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

  def reset(self):
    ob = self.env.reset()
    for _ in range(self.k):
      self.frames.append(ob)
    return self._get_ob()

  def step(self, action):
    ob, reward, done, info = self.env.step(action)
    self.frames.append(ob)
    return self._get_ob(), reward, done, info

  def _get_ob(self):
    assert len(self.frames) == self.k
    return LazyFrames(list(self.frames))


class MaxAndSkipEnv(gym.Wrapper):
  """Max and skip env."""

  def __init__(self, env, skip=4):
    """Return only every `skip`-th frame."""
    gym.Wrapper.__init__(self, env)
    # Most recent raw observations (for max pooling across time steps).
    self._obs_buffer = np.zeros((2,) + env.observation_space.shape,
                                dtype=np.uint8)
    self._skip = skip

  def reset(self, **kwargs):
    return self.env.reset(**kwargs)

  def step(self, action):
    """Repeat action, sum reward, and max over last observations."""
    total_reward = 0.0
    done = None
    for i in range(self._skip):
      obs, reward, done, info = self.env.step(action)
      if i == self._skip - 2: self._obs_buffer[0] = obs
      if i == self._skip - 1: self._obs_buffer[1] = obs
      total_reward += reward
      if done:
        break
    # Note that the observation on the done=True frame
    # doesn't matter
    max_frame = self._obs_buffer.max(axis=0)

    return max_frame, total_reward, done, info


def wrap_atari(env, warp=False, frame_skip=False, frame_stack=False):
  if warp:
    env = WarpFrame(env)
  if frame_skip:
    env = MaxAndSkipEnv(env, frame_skip)
  if frame_stack:
    env = FrameStack(env, frame_stack)
  return env
