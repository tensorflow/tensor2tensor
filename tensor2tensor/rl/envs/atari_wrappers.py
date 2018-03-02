# Copied from baselines
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

# Various wrappers copied from Baselines
import gym
import numpy as np
from collections import deque
import gym
from gym import spaces


class WarpFrame(gym.ObservationWrapper):
  def __init__(self, env):
    """Warp frames to 84x84 as done in the Nature paper and later work."""
    gym.ObservationWrapper.__init__(self, env)
    self.width = 84
    self.height = 84
    self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(self.height, self.width, 1), dtype=np.uint8)

  def observation(self, frame):
    import cv2
    cv2.ocl.setUseOpenCL(False)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
    return frame[:, :, None]

class LazyFrames(object):
  def __init__(self, frames):
    """This object ensures that common frames between the observations are only stored once.
    It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
    buffers.
    This object should only be converted to numpy array before being passed to the model.
    You'd not believe how complex the previous solution was."""
    self._frames = frames

  def __array__(self, dtype=None):
    out = np.concatenate(self._frames, axis=2)
    if dtype is not None:
      out = out.astype(dtype)
    return out

class FrameStack(gym.Wrapper):
  def __init__(self, env, k):
    """Stack k last frames.
    Returns lazy array, which is much more memory efficient.
    See Also
    --------
    baselines.common.atari_wrappers.LazyFrames
    """
    gym.Wrapper.__init__(self, env)
    self.k = k
    self.frames = deque([], maxlen=k)
    shp = env.observation_space.shape
    self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

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
  def __init__(self, env, skip=4):
    """Return only every `skip`-th frame"""
    gym.Wrapper.__init__(self, env)
    # most recent raw observations (for max pooling across time steps)
    self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
    self._skip = skip

  def reset(self):
    return self.env.reset()

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

  def reset(self, **kwargs):
    return self.env.reset(**kwargs)

def wrap_atari(env, warp=False, frame_skip=False, frame_stack=False):
  if warp:
    env = WarpFrame(env)
  if frame_skip:
    env = MaxAndSkipEnv(env, frame_skip)
  if frame_stack:
    env = FrameStack(env, frame_stack)
  return env
