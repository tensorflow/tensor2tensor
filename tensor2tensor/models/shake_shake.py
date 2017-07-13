from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.models import common_hparams
from tensor2tensor.models import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf

def shake_shake_block_branch(x, conv_filters, stride):
  x = tf.nn.relu(x)
  x = common_layers.conv(x, conv_filters, (3, 3), (stride, stride))
  x = tf.layers.batch_normalization(x)
  x = tf.nn.relu(x)
  x = common_layers.conv(x, conv_filters, (3, 3), (1, 1))
  x = tf.layers.batch_normalization(x)
  return x


def downsampling_residual_branch(x, conv_filters):
  x = tf.nn.relu(x)

  x1 = tf.layers.average_pooling2d(x, pool_size=(1, 1), strides=(2, 2))
  x1 = common_layers.conv(x1, conv_filters / 2, (1, 1))

  x2 = tf.pad(x[:, 1:, 1:], [[0, 0], [0, 1], [0, 1], [0, 0]])
  x2 = tf.layers.average_pooling2d(x2, pool_size=(1, 1), strides=(2, 2))
  x2 = common_layers.conv(x2, conv_filters / 2, (1, 1))

  return tf.concat([x1, x2], axis=3)


def shake_shake_block(x, conv_filters, stride):
  branch1 = shake_shake_block_branch(x, conv_filters, stride)
  branch2 = shake_shake_block_branch(x, conv_filters, stride)
  if x.shape[-1] == conv_filters:
    skip = tf.identity(x)
  else:
    skip = downsampling_residual_block(x)

  # TODO(rshin): Set equal=true when testing.
  # TODO(rshin): Use different alpha for each image in batch.
  return skip + common_layers.shakeshake2(branch1, branch2)


def shake_shake_stage(x, num_blocks, conv_filters, initial_stride):
  x = shake_shake_block(x, conv_filters, initial_stride)
  for _ in xrange(num_blocks - 1):
    x = shake_shake_block(x, conv_filters, 1)
  return x


@registry.register_model
class ShakeShake(t2t_model.T2TModel):

  def model_fn_body(self, features):
    hparams = self._hparams

    inputs = features["inputs"]
    assert (hparams.num_hidden_layers - 2) % 6 == 0
    blocks_per_stage = (hparams.num_hidden_layers - 2) / 6

    # For canonical Shake-Shake, the entry flow is a 3x3 convolution with 16
    # filters then a batch norm. Instead we use the one in SmallImageModality,
    # which also seems to include a layer norm.
    x = inputs
    with tf.name_scope('shake_shake_stage_1'):
      x = shake_shake_stage(x, hparams.base_filters, blocks_per_stage)
    with tf.name_scope('shake_shake_stage_2'):
      x = shake_shake_stage(x, hparams.base_filters * 2, blocks_per_stage)
    with tf.name_scope('shake_shake_stage_3'):
      x = shake_shake_stage(x, hparams.base_filters * 4, blocks_per_stage)

    # For canonical Shake-Shake, we should perform 8x8 average pooling and then
    # have a fully-connected layer (which produces the logits for each class).
    # Instead, we just use the Xception exit flow in ClassLabelModality.
    return x

@registry.register_hparams
def shakeshake_cifar10():
  hparams = common_hparams.basic_params1()
  # This leads to effective batch size 128 when number of GPUs is 2
  hparams.batch_size = 4096 * 4
  hparams.hidden_size = 16
  hparams.dropout = 0
  hparams.label_smoothing = 0.0
  hparams.clip_grad_norm = 2.0
  hparams.num_hidden_layers = 26
  hparams.kernel_height = -1  # Unused
  hparams.kernel_width = -1  # Unused
  hparams.learning_rate_decay_scheme = "cosine"
  # Model should be run for 700000 steps with batch size 128 (~1800 epochs)
  hparams.learning_rate_cosine_cycle_steps = 700000
  hparams.learning_rate = 0.2
  hparams.learning_rate_warmup_steps = 3000
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 0.1  # Effective value should be ~1e-4
  hparams.optimizer = "Momentum"
  hparams.optimizer_momentum_momentum = 0.9
  hparams.add_hparam('base_filters', 16)
  return hparams
