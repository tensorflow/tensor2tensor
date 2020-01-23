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

"""Conditional FVD metric on video.

FVD - Frechet Video Distance

This is the metric that is inspired by FID, but applied to
video rather than to images.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


class VideoEvaluationDataset(
    collections.namedtuple(
        'VideoEvaluationDataset',
        ['n_input_frames', 'n_output_frames', 'get_video_batch_fn'])):
  """Dataset for video evaluation.

  This tuple describes the video problem for Evaluation.
  Args:
     n_input_frames: number of frames passed to the model to condition on.
     n_output_frames: number of frames that model should return.
     get_video_batch_fn: function that accepts a batch size and returns a tensor
       with real video, which should match <uint8>[batch_size, N, height, width,
       depth], where N is n_input_frames + n_output_frames.
  """
  pass


class Model(
    collections.namedtuple('Model', [
        'apply_fn', 'load_fn',
    ])):
  """Model that should be evaluated.

  Args:
    apply_fn: will be called with a single tensor (floats between 0 and 255
              of shape [batch_size, n_input_frames, height, width, depth]),
              that will contain input frames.
              it should return a single tensor with output frames (floats
              between 0 and 255, of shape
              [batch_size, n_output_frames, height, width, depth])
    load_fn: Callable, that receives session as an argument.
             Should load the variables from the checkpoint.
  """
  pass


def evaluate_model(video_eval_dataset, model, num_batches, batch_size):
  """Computes the FVD video metric.

  Args:
    video_eval_dataset: VideoEvaluationDataset tuple with video and frames
      information.
    model: Model tuple with model to evaluate.
    num_batches: number of batches to evaluate.
    batch_size: number of videos to compute per batch.

  Returns:
    FVD metric (float).
  """
  del video_eval_dataset, model, num_batches, batch_size
