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
"""Traditional Student-Teacher Distillation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.layers import common_hparams
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


@registry.register_model
class Distillation(t2t_model.T2TModel):
  """Distillation from a teacher to student network.

  First, a teacher is train on a task; Second, a student is trained to perform
  the task while matching the teacher's softened outputs. For more details, see
  the paper below.

  In the hparams passed to this model include the desired
  {teacher/student}_model and {teacher/student}_hparams to be used. Also,
  specify the distillation temperature and task-distillation balance.

  Distilling the Knowledge in a Neural Network
  Hinton, Vinyals and Dean
  https://arxiv.org/abs/1503.02531
  """

  def __init__(self,
               hparams,
               mode=tf.estimator.ModeKeys.TRAIN,
               problem_hparams=None,
               data_parallelism=None,
               decode_hparams=None):
    assert hparams.distill_phase in ["train", "distill"]

    if hparams.distill_phase == "train" and hparams.teacher_learning_rate:
      hparams.learning_rate = hparams.teacher_learning_rate
    elif hparams.distill_phase == "distill" and hparams.student_learning_rate:
      hparams.learning_rate = hparams.student_learning_rate

    self.teacher_hparams = registry.hparams(hparams.teacher_hparams)
    self.teacher_model = registry.model(
        hparams.teacher_model)(self.teacher_hparams, mode, problem_hparams,
                               data_parallelism, decode_hparams)
    self.student_hparams = registry.hparams(hparams.student_hparams)
    self.student_model = registry.model(
        hparams.student_model)(self.student_hparams, mode, problem_hparams,
                               data_parallelism, decode_hparams)
    super(Distillation, self).__init__(hparams, mode, problem_hparams,
                                       data_parallelism, decode_hparams)

  def body(self, features):
    hp = self.hparams
    is_distill = hp.distill_phase == "distill"

    targets = features["targets_raw"]
    targets = tf.squeeze(targets, [1, 2, 3])
    one_hot_targets = tf.one_hot(targets, hp.num_classes, dtype=tf.float32)

    # Teacher Network
    with tf.variable_scope("teacher"):
      teacher_outputs = self.teacher_model.body(features)
      tf.logging.info("teacher output shape: %s" % teacher_outputs.get_shape())
      teacher_outputs = tf.reduce_mean(teacher_outputs, axis=[1, 2])
      teacher_logits = tf.layers.dense(teacher_outputs, hp.num_classes)

      teacher_task_xent = tf.nn.softmax_cross_entropy_with_logits_v2(
          labels=one_hot_targets, logits=teacher_logits)
      outputs = teacher_logits

    if is_distill:
      # Load teacher weights
      tf.train.init_from_checkpoint(hp.teacher_dir, {"teacher/": "teacher/"})
      # Do not train the teacher
      trainable_vars = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
      del trainable_vars[:]

    # Student Network
    if is_distill:
      with tf.variable_scope("student"):
        student_outputs = self.student_model.body(features)
        tf.logging.info(
            "student output shape: %s" % student_outputs.get_shape())
        student_outputs = tf.reduce_mean(student_outputs, axis=[1, 2])
        student_logits = tf.layers.dense(student_outputs, hp.num_classes)

        student_task_xent = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=one_hot_targets, logits=student_logits)
        teacher_targets = tf.nn.softmax(teacher_logits / hp.distill_temperature)
        student_distill_xent = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.stop_gradient(teacher_targets), logits=student_logits)

        outputs = student_logits

        # Summaries
        tf.summary.scalar("distill_xent", student_distill_xent)

    if not is_distill:
      phase_loss = teacher_task_xent
    else:
      phase_loss = hp.task_balance * student_task_xent
      phase_loss += (1 - hp.task_balance) * student_distill_xent

    losses = {"training": phase_loss}
    outputs = tf.reshape(outputs, [-1, 1, 1, 1, outputs.shape[1]])

    return outputs, losses

  def top(self, body_output, features):
    return body_output


def distill_base():
  """Set of hyperparameters."""
  # Base
  hparams = common_hparams.basic_params1()

  # teacher/student parameters
  hparams.add_hparam("teacher_model", "")
  hparams.add_hparam("teacher_hparams", "")
  hparams.add_hparam("student_model", "")
  hparams.add_hparam("student_hparams", "")

  # Distillation parameters
  # WARNING: distill_phase hparam will be overwritten in /bin/t2t_distill.py
  hparams.add_hparam("distill_phase", None)
  hparams.add_hparam("task_balance", 1.0)
  hparams.add_hparam("distill_temperature", 1.0)
  hparams.add_hparam("num_classes", 10)

  # Optional Phase-specific hyperparameters
  hparams.add_hparam("teacher_learning_rate", None)
  hparams.add_hparam("student_learning_rate", None)

  # Training parameters (stolen from ResNet)
  hparams.batch_size = 128
  hparams.optimizer = "Momentum"
  hparams.optimizer_momentum_momentum = 0.9
  hparams.optimizer_momentum_nesterov = True
  hparams.weight_decay = 1e-4
  hparams.clip_grad_norm = 0.0
  # (base_lr=0.1) * (batch_size=128*8 (on TPU, or 8 GPUs)=1024) / (256.)
  hparams.learning_rate = 0.4
  hparams.learning_rate_decay_scheme = "cosine"
  # For image_imagenet224, 120k training steps, which effectively makes this a
  # cosine decay (i.e. no cycles).
  hparams.learning_rate_cosine_cycle_steps = 120000
  hparams.initializer = "normal_unit_scaling"
  hparams.initializer_gain = 2.

  return hparams


@registry.register_hparams
def distill_resnet_32_to_15_cifar20x5():
  """Set of hyperparameters."""
  hparams = distill_base()
  hparams.teacher_model = "resnet"
  hparams.teacher_hparams = "resnet_cifar_32"
  hparams.student_model = "resnet"
  hparams.student_hparams = "resnet_cifar_15"

  hparams.optimizer_momentum_nesterov = True
  # (base_lr=0.1) * (batch_size=128*8 (on TPU, or 8 GPUs)=1024) / (256.)
  hparams.teacher_learning_rate = 0.25 * 128. * 8. / 256.
  hparams.student_learning_rate = 0.2 * 128. * 8. / 256.
  hparams.learning_rate_decay_scheme = "piecewise"
  hparams.add_hparam("learning_rate_boundaries", [40000, 60000, 80000])
  hparams.add_hparam("learning_rate_multiples", [0.1, 0.01, 0.001])

  hparams.task_balance = 0.28
  hparams.distill_temperature = 2.0

  hparams.num_classes = 20

  return hparams
