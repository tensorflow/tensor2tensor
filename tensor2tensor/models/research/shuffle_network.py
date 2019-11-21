from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensor2tensor.layers import common_hparams
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model


def ror(x, n, p=1):
    """Bitwise rotation right"""
    a = tf.bitwise.right_shift(x, p)
    b = tf.bitwise.left_shift(1, p) - 1
    c = tf.bitwise.bitwise_and(x, b)
    d = tf.bitwise.left_shift(c, n - p)

    return a + d


def rol(x, n, p=1):
    """Bitwise rotation left"""
    a = tf.bitwise.left_shift(x, p)
    b = tf.bitwise.left_shift(1, n) - 1
    c = tf.bitwise.bitwise_and(a, b)
    d = tf.bitwise.right_shift(x, n - p)
    return tf.bitwise.bitwise_or(c, d)


def shuffle_layer(inputs, shuffle_fn=rol):
    """Shuffles the elements according to bitwise left or right rotation on their indices"""
    length = tf.shape(inputs)[1]
    n_bits = tf.log(tf.cast(length - 1, tf.float32)) / tf.log(2.0)
    n_bits = tf.cast(n_bits, tf.int32) + 1

    indices = tf.range(0, length)
    rev_indices = shuffle_fn(indices, n_bits)
    return tf.gather(inputs, rev_indices, axis=1)


def reverse_shuffle_layer(inputs):
    return shuffle_layer(inputs, ror)


def conv_linear_map(inputs, nin, nout, bias_start, prefix):
    with tf.variable_scope(prefix):
        inp_shape = tf.shape(inputs)

        initializer = tf.variance_scaling_initializer(scale=1.0, mode="fan_avg", distribution="uniform")
        kernel = tf.get_variable("CvK", [nin, nout], initializer=initializer)
        bias_term = tf.get_variable("CvB", [nout], initializer=tf.constant_initializer(0.0))

        res = tf.matmul(tf.reshape(inputs, [inp_shape[0] * inp_shape[1], nin]), kernel)
        res = tf.reshape(res, [inp_shape[0], inp_shape[1], nout])
        return res + bias_start + bias_term


class SwitchLayer:

    def __init__(self, prefix, dropout, mode) -> None:
        self.prefix = prefix
        self.dropout = dropout
        self.mode = mode
        self.batch_size = None
        self.length = None
        self.num_units = None
        self.n_bits = None

    def linear_map(self, inputs, suffix, bias_start, in_units, out_units):
        """
        2 input to 2 output linear map
        """
        inputs = tf.reshape(inputs, [self.batch_size, self.length // 2, in_units * 2])
        res = conv_linear_map(inputs, in_units * 2, out_units * 2, bias_start, self.prefix + "/" + suffix)
        return tf.reshape(res, [self.batch_size, self.length, out_units])

    def gated_linear_map(self, inputs, suffix, bias_start_reset, in_units, out_units):
        """
        Linear mapping with two reset gates
        """

        def reset_gate(name):
            prefix = self.prefix + name + suffix
            reset = conv_linear_map(inputs, in_units * 2, in_units * 2, bias_start_reset, prefix)
            return tf.nn.sigmoid(reset)

        inputs = tf.reshape(inputs, [self.batch_size, self.length // 2, in_units * 2])

        reset1 = reset_gate("/reset1/")
        reset2 = reset_gate("/reset2/")
        res1 = conv_linear_map(inputs * reset1, in_units * 2, out_units, 0.0, self.prefix + "/cand1/" + suffix)
        res2 = conv_linear_map(inputs * reset2, in_units * 2, out_units, 0.0, self.prefix + "/cand2/" + suffix)

        res = tf.concat([res1, res2], axis=2)
        res = tf.reshape(res, [self.batch_size, self.length, out_units])
        return tf.nn.tanh(res)

    def __call__(self, inputs, residual_inputs):
        input_shape = tf.shape(inputs)
        self.batch_size = input_shape[0]
        self.length = input_shape[1]
        self.num_units = inputs.shape.as_list()[2]

        self.n_bits = tf.log(tf.cast(self.length - 1, tf.float32)) / tf.log(2.0)
        self.n_bits = tf.floor(self.n_bits) + 1

        initializer = tf.constant_initializer(0.5)
        residual_scale = tf.get_variable(self.prefix + "/residual_scale", [self.num_units], initializer=initializer)

        shuffled_input = self.shuffle_inputs(inputs)
        mem_all = inputs + residual_inputs * residual_scale

        # calculate the new value
        candidate = self.gated_linear_map(mem_all, "c", 0.5, self.num_units, self.num_units)
        gate = tf.nn.sigmoid(self.linear_map(mem_all, "g", 0.5, self.num_units, self.num_units))

        candidate = gate * shuffled_input + (1 - gate) * candidate

        if self.dropout > 0:
            candidate = tf.nn.dropout(candidate, rate=self.dropout / self.n_bits)
        if not self.dropout == 0.0 and self.mode == tf.estimator.ModeKeys.TRAIN:
            candidate = candidate * tf.random_normal(tf.shape(candidate), mean=1.0, stddev=0.001)

        return candidate

    def shuffle_inputs(self, inputs):
        x = tf.range(0, self.length)
        xor_indices = tf.bitwise.bitwise_xor(x, 1)
        input_xor = tf.gather(inputs[:, :, :self.num_units // 2], xor_indices, axis=1)
        return tf.concat([input_xor, inputs[:, :, self.num_units // 2:]], axis=2)


def shuffle_network(inputs, hparams):
    """Neural Benes Network with skip connections between blocks."""

    def forward_step(state, layer_nr):
        with tf.variable_scope("forward"):
            last_state, residuals = state
            prev = residuals[layer_nr, :, :, :]
            cur = SwitchLayer("switch", hparams.dropout, hparams.mode)(last_state, prev)
            return shuffle_layer(cur), residuals

    def reverse_step(state, layer_nr):
        with tf.variable_scope("reverse"):
            last_state, residuals = state
            prev = residuals[layer_nr, :, :, :]
            cur = SwitchLayer("reverse_switch", hparams.dropout, hparams.mode)(last_state, prev)
            return reverse_shuffle_layer(cur), residuals

    input_shape = tf.shape(inputs)
    n_bits = tf.log(tf.cast(input_shape[1] - 1, tf.float32)) / tf.log(2.0)
    n_bits = tf.cast(n_bits, tf.int32) + 1

    residuals_queue = tf.zeros([n_bits * 2, input_shape[0], input_shape[1], input_shape[2]])
    block_out = tf.tanh(inputs)

    for k in range(hparams.num_hidden_layers):
        with tf.variable_scope("benes_block_" + str(k), reuse=tf.AUTO_REUSE):
            forward_outputs, _ = tf.scan(
                forward_step,
                tf.range(0, n_bits),
                initializer=(block_out, residuals_queue),
                parallel_iterations=1,
                swap_memory=True
            )

            forward_outputs = tf.concat([tf.expand_dims(block_out, axis=0), forward_outputs], axis=0)
            forward_last = forward_outputs[-1, :, :, :]

            reverse_outputs, _ = tf.scan(
                reverse_step,
                tf.range(n_bits, n_bits * 2),
                initializer=(forward_last, residuals_queue),
                parallel_iterations=1,
                swap_memory=True
            )

            block_out = reverse_outputs[-1, :, :, :]
            residuals_queue = tf.concat([forward_outputs, reverse_outputs], axis=0)

    return SwitchLayer("last_layer", hparams.dropout, hparams.mode)(block_out, residuals_queue[n_bits * 2, :, :, :])


@registry.register_model
class ShuffleNetwork(t2t_model.T2TModel):
    """
    Implementation of "Neural Shuffle-Exchange Networks − Sequence Processing in O(n log n) Time" paper
    by K.Freivalds, E.Ozolins, A.Sostaks.
    Paper: https://papers.nips.cc/paper/8889-neural-shuffle-exchange-networks-sequence-processing-in-on-log-n-time.pdf
    """

    def bottom(self, features):
        inputs = features["inputs"]
        targets = features["targets"]
        inputs_length = tf.shape(inputs)[1]
        targets_length = tf.shape(targets)[1]

        length = tf.maximum(inputs_length, targets_length)
        p = tf.log(tf.cast(length, tf.float32)) / tf.log(2.0)
        p = tf.cast(tf.ceil(p), tf.int32)
        pad_len = tf.pow(2, p)

        features["inputs"] = tf.pad(inputs, [[0, 0], [0, pad_len - inputs_length], [0, 0], [0, 0]])
        features["targets"] = tf.pad(targets, [[0, 0], [0, pad_len - targets_length], [0, 0], [0, 0]])
        return super().bottom(features)

    def loss(self, logits, features):
        onehot_labels = tf.one_hot(features["targets"], self._problem_hparams.vocab_size["targets"])
        cost_vector = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=onehot_labels)
        return tf.reduce_mean(cost_vector)

    def body(self, features):
        inputs = tf.squeeze(features["inputs"], axis=2)
        logits = shuffle_network(inputs, self._hparams)
        return tf.expand_dims(logits, axis=2)


@registry.register_hparams
def shuffle_network_baseline():
    """Set of hyperparameters."""
    hparams = common_hparams.basic_params1()
    hparams.hidden_size = 48 * 4  # feature maps
    hparams.num_hidden_layers = 1  # block count

    hparams.clip_grad_norm = 0.  # no gradient clipping

    hparams.optimizer = "adam"
    hparams.optimizer_adam_epsilon = 1e-5
    hparams.learning_rate_schedule = "legacy"
    hparams.learning_rate_decay_scheme = "noam"
    hparams.learning_rate = 0.1
    hparams.initializer_gain = 1.0
    hparams.initializer = "uniform_unit_scaling"
    hparams.optimizer_adam_beta1 = 0.9
    hparams.optimizer_adam_beta2 = 0.999

    hparams.dropout = 0.1
    hparams.label_smoothing = 0.
    hparams.weight_decay = 0.

    return hparams
