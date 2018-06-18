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
"""Utilities for using batched environments."""

# The code was based on Danijar Hafner's code from tf.agents:
# https://github.com/tensorflow/agents/blob/master/agents/tools/wrappers.py
# https://github.com/tensorflow/agents/blob/master/agents/scripts/utility.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import multiprocessing
import os
import random
import signal
import subprocess
import sys
import traceback
import gym

from tensor2tensor.rl.envs import batch_env
from tensor2tensor.rl.envs import py_func_batch_env
from tensor2tensor.rl.envs import simulated_batch_env

import tensorflow as tf


class EvalVideoWrapper(gym.Wrapper):
  """Wrapper for recording videos during eval phase.

  This wrapper is designed to record videos via gym.wrappers.Monitor and
  simplifying its usage in t2t collect phase.
  It alleviate the limitation of Monitor, which doesn't allow reset on an
  active environment.

  EvalVideoWrapper assumes that only every second trajectory (after every
  second reset) will be used by the caller:
  - on the "active" runs it behaves as gym.wrappers.Monitor,
  - on the "inactive" runs it doesn't call underlying environment and only
    returns last seen observation.
  Videos are only generated during the active runs.
  """

  def __init__(self, env):
    super(EvalVideoWrapper, self).__init__(env)
    self._reset_counter = 0
    self._active = False
    self._last_returned = None

  def _step(self, action):
    if self._active:
      self._last_returned = self.env.step(action)
    if self._last_returned is None:
      raise Exception("Environment stepped before proper reset.")
    return self._last_returned

  def _reset(self, **kwargs):
    self._reset_counter += 1
    if self._reset_counter % 2 == 1:
      self._active = True
      return self.env.reset(**kwargs)

    self._active = False
    self._last_returned = (self._last_returned[0],
                           self._last_returned[1],
                           False,  # done = False
                           self._last_returned[3])
    return self._last_returned[0]


class ExternalProcessEnv(object):
  """Step environment in a separate process for lock free parallelism."""

  # Message types for communication via the pipe.
  _ACCESS = 1
  _CALL = 2
  _RESULT = 3
  _EXCEPTION = 4
  _CLOSE = 5

  def __init__(self, constructor, xvfb):
    """Step environment in a separate process for lock free parallelism.

    The environment will be created in the external process by calling the
    specified callable. This can be an environment class, or a function
    creating the environment and potentially wrapping it. The returned
    environment should not access global variables.

    Args:
      constructor: Callable that creates and returns an OpenAI gym environment.
      xvfb:  Frame buffer.

    Attributes:
      observation_space: The cached observation space of the environment.
      action_space: The cached action space of the environment.
    """
    self._conn, conn = multiprocessing.Pipe()
    if xvfb:
      server_id = random.randint(10000, 99999)
      auth_file_id = random.randint(10000, 99999999999)

      xauthority_path = "/tmp/Xauthority_{}".format(auth_file_id)

      command = "Xvfb :{} -screen 0 1400x900x24 -nolisten tcp -auth {}".format(
          server_id, xauthority_path)
      with open(os.devnull, "w") as devnull:
        proc = subprocess.Popen(command.split(), shell=False, stdout=devnull,
                                stderr=devnull)
        atexit.register(lambda: os.kill(proc.pid, signal.SIGKILL))

      def constructor_using_xvfb():
        os.environ["DISPLAY"] = ":{}".format(server_id)
        os.environ["XAUTHORITY"] = xauthority_path
        return constructor()

      self._process = multiprocessing.Process(
          target=self._worker, args=(constructor_using_xvfb, conn))
    else:
      self._process = multiprocessing.Process(
          target=self._worker, args=(constructor, conn))

    atexit.register(self.close)
    self._process.start()
    self._observ_space = None
    self._action_space = None

  @property
  def observation_space(self):
    if not self._observ_space:
      self._observ_space = self.__getattr__("observation_space")
    return self._observ_space

  @property
  def action_space(self):
    if not self._action_space:
      self._action_space = self.__getattr__("action_space")
    return self._action_space

  def __getattr__(self, name):
    """Request an attribute from the environment.

    Note that this involves communication with the external process, so it can
    be slow.

    Args:
      name: Attribute to access.

    Returns:
      Value of the attribute.
    """
    self._conn.send((self._ACCESS, name))
    return self._receive()

  def call(self, name, *args, **kwargs):
    """Asynchronously call a method of the external environment.

    Args:
      name: Name of the method to call.
      *args: Positional arguments to forward to the method.
      **kwargs: Keyword arguments to forward to the method.

    Returns:
      Promise object that blocks and provides the return value when called.
    """
    payload = name, args, kwargs
    self._conn.send((self._CALL, payload))
    return self._receive

  def close(self):
    """Send a close message to the external process and join it."""
    try:
      self._conn.send((self._CLOSE, None))
      self._conn.close()
    except IOError:
      # The connection was already closed.
      pass
    self._process.join()

  def step(self, action, blocking=True):
    """Step the environment.

    Args:
      action: The action to apply to the environment.
      blocking: Whether to wait for the result.

    Returns:
      Transition tuple when blocking, otherwise callable that returns the
      transition tuple.
    """
    promise = self.call("step", action)
    if blocking:
      return promise()
    return promise

  def reset(self, blocking=True):
    """Reset the environment.

    Args:
      blocking: Whether to wait for the result.

    Returns:
      New observation when blocking, otherwise callable that returns the new
      observation.
    """
    promise = self.call("reset")
    if blocking:
      return promise()
    return promise

  def _receive(self):
    """Wait for a message from the worker process and return its payload.

    Raises:
      Exception: An exception was raised inside the worker process.
      KeyError: The received message is of an unknown type.

    Returns:
      Payload object of the message.
    """
    message, payload = self._conn.recv()
    # Re-raise exceptions in the main process.
    if message == self._EXCEPTION:
      stacktrace = payload
      raise Exception(stacktrace)
    if message == self._RESULT:
      return payload
    raise KeyError("Received message of unexpected type {}".format(message))

  def _worker(self, constructor, conn):
    """The process waits for actions and sends back environment results.

    Args:
      constructor: Constructor for the OpenAI Gym environment.
      conn: Connection for communication to the main process.
    """
    try:
      env = constructor()
      while True:
        try:
          # Only block for short times to have keyboard exceptions be raised.
          if not conn.poll(0.1):
            continue
          message, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
          break
        if message == self._ACCESS:
          name = payload
          result = getattr(env, name)
          conn.send((self._RESULT, result))
          continue
        if message == self._CALL:
          name, args, kwargs = payload
          result = getattr(env, name)(*args, **kwargs)
          conn.send((self._RESULT, result))
          continue
        if message == self._CLOSE:
          assert payload is None
          env.close()
          break
        raise KeyError("Received message of unknown type {}".format(message))
    except Exception:  # pylint: disable=broad-except
      stacktrace = "".join(traceback.format_exception(*sys.exc_info()))  # pylint: disable=no-value-for-parameter
      tf.logging.error("Error in environment process: {}".format(stacktrace))
      conn.send((self._EXCEPTION, stacktrace))
    conn.close()


def batch_env_factory(hparams, xvfb=False):
  """Factory of batch envs."""

  environment_spec = hparams.environment_spec

  if environment_spec.simulated_env:
    # TODO(piotrmilos): Consider passing only relevant paramters
    cur_batch_env = _define_simulated_batch_env(
      hparams, hparams.num_agents, hparams.problem,
        hparams.simulation_random_starts,
        hparams.intrinsic_reward_scale)
  else:

    cur_batch_env = _define_batch_env(hparams.environment_spec, hparams.num_agents,
                                      xvfb=xvfb)
  return cur_batch_env


def _define_batch_env(environment_spec, num_agents, xvfb=False):
  """Create environments and apply all desired wrappers."""

  with tf.variable_scope("environments"):
    envs = [
        ExternalProcessEnv(environment_spec.env_lambda, xvfb)
        for _ in range(num_agents)]
    env = batch_env.BatchEnv(envs, blocking=False)
    env = py_func_batch_env.PyFuncBatchEnv(env)
    return env


def _define_simulated_batch_env(hparams, num_agents, problem,
                               simulation_random_starts=False,
                               intrinsic_reward_scale=0.):
  cur_batch_env = simulated_batch_env.SimulatedBatchEnv(
      hparams, num_agents, problem, simulation_random_starts,
      intrinsic_reward_scale)
  return cur_batch_env


def get_action_space(environment_spec):
  """Get action spece associated with environment spec
    
  Args:
     environment_spec:  EnvironmentSpec object
     
  Returns:
    OpenAi Gym action space
  """
  action_space = environment_spec.env_lambda().action_space
  action_shape = list(parse_shape(action_space))
  action_dtype = parse_dtype(action_space)

  return action_space, action_shape, action_dtype


def get_policy(observations, hparams):
  """Get policy network
  
  Args:
    observations: Tensor with observations
    hparams: parameters 
    
  Returns:
    Tensor with policy and value function output
  """
  policy_network_lambda = hparams.policy_network
  action_space, _, _ = get_action_space(hparams.environment_spec)
  return policy_network_lambda(action_space, hparams, observations)


def parse_shape(space):
  """Get a tensor shape from a OpenAI Gym space.

  Args:
    space: Gym space.

  Returns:
    Shape tuple.
  """
  if isinstance(space, gym.spaces.Discrete):
    return ()
  if isinstance(space, gym.spaces.Box):
    return space.shape
  raise NotImplementedError()

def parse_dtype(space):
  """Get a tensor dtype from a OpenAI Gym space.

  Args:
    space: Gym space.

  Returns:
    TensorFlow data type.
  """
  if isinstance(space, gym.spaces.Discrete):
    return tf.int32
  if isinstance(space, gym.spaces.Box):
    return tf.float32
  raise NotImplementedError()