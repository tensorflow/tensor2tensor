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

"""Base class for env problems with RGB array as observation space."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import png
import six
from tensor2tensor.data_generators import video_utils
from tensor2tensor.envs import env_problem
from tensor2tensor.envs import gym_env_problem
from tensor2tensor.utils import contrib
import tensorflow.compat.v1 as tf

_IMAGE_ENCODED_FIELD = "image/encoded"
_IMAGE_FORMAT_FIELD = "image/format"
_IMAGE_HEIGHT_FIELD = "image/height"
_IMAGE_WIDTH_FIELD = "image/width"
_FRAME_NUMBER_FIELD = "frame_number"

_FORMAT = "png"


class RenderedEnvProblem(gym_env_problem.GymEnvProblem,
                         video_utils.VideoProblem):
  """An `EnvProblem` when observations are RGB arrays.

  This takes care of wrapping a rendered gym environment to behave like a
  `VideoProblem`. This class assumes the underlying gym environment is either a
  `gym_utils.RenderedEnv` or it natively returns rendered scene for
  observations. i.e. the underlying gym environment should have a
  `Box` observation space with the following shape: [frame_height, frame_width,
  channels]

  Note: The method resolution order for this class is:
  `RenderedEnvProblem`, `EnvProblem`, `Env`, `VideoProblem`, `Problem`
  """

  def __init__(self, *args, **kwargs):
    """Initialize by calling both parents' constructors."""
    gym_env_problem.GymEnvProblem.__init__(self, *args, **kwargs)
    video_utils.VideoProblem.__init__(self)

  def initialize_environments(self,
                              batch_size=1,
                              parallelism=1,
                              rendered_env=True,
                              per_env_kwargs=None,
                              **kwargs):
    gym_env_problem.GymEnvProblem.initialize_environments(
        self, batch_size=batch_size, parallelism=parallelism,
        per_env_kwargs=per_env_kwargs, **kwargs)
    # Assert the underlying gym environment has correct observation space
    if rendered_env:
      assert len(self.observation_spec.shape) == 3

  def example_reading_spec(self):
    """Return a mix of env and video data fields and decoders."""
    slim = contrib.slim()
    video_fields, video_decoders = (
        video_utils.VideoProblem.example_reading_spec(self))
    env_fields, env_decoders = (
        gym_env_problem.GymEnvProblem.example_reading_spec(self))

    # Remove raw observations field since we want to capture them as videos.
    env_fields.pop(env_problem.OBSERVATION_FIELD)
    env_decoders.pop(env_problem.OBSERVATION_FIELD)

    # Add frame number spec and decoder.
    env_fields[_FRAME_NUMBER_FIELD] = tf.FixedLenFeature((1,), tf.int64)
    env_decoders[_FRAME_NUMBER_FIELD] = slim.tfexample_decoder.Tensor(
        _FRAME_NUMBER_FIELD)

    # Add video fields and decoders
    env_fields.update(video_fields)
    env_decoders.update(video_decoders)
    return env_fields, env_decoders

  def _generate_time_steps(self, trajectory_list):
    """Transforms time step observations to frames of a video."""
    for time_step in gym_env_problem.GymEnvProblem._generate_time_steps(
        self, trajectory_list):
      # Convert the rendered observations from numpy to png format.
      frame_np = np.array(time_step.pop(env_problem.OBSERVATION_FIELD))
      frame_np = frame_np.reshape(
          [self.frame_height, self.frame_width, self.num_channels])
      # TODO(msaffar) Add support for non RGB rendered environments
      frame = png.from_array(frame_np, "RGB", info={"bitdepth": 8})
      frame_buffer = six.BytesIO()
      frame.save(frame_buffer)

      # Put the encoded frame back.
      time_step[_IMAGE_ENCODED_FIELD] = [frame_buffer.getvalue()]
      time_step[_IMAGE_FORMAT_FIELD] = [_FORMAT]
      time_step[_IMAGE_HEIGHT_FIELD] = [self.frame_height]
      time_step[_IMAGE_WIDTH_FIELD] = [self.frame_width]

      # Add the frame number
      time_step[_FRAME_NUMBER_FIELD] = time_step[env_problem.TIMESTEP_FIELD]
      yield time_step

  @property
  def num_channels(self):
    return self.observation_spec.shape[2]

  @property
  def frame_height(self):
    return self.observation_spec.shape[0]

  @property
  def frame_width(self):
    return self.observation_spec.shape[1]

  @property
  def total_number_of_frames(self):
    """Upper bound on the total number of frames across all environments.

    This is used to decide sharding. See `VideoProblem.total_number_of_frames`
    for more details.

    Returns:
      number of frames among all examples in the dataset.
    """
    return self.trajectories.num_time_steps
