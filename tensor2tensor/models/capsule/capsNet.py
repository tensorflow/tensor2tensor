#encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf

from tensor2tensor.models.capsule.capsLayer import CapsLayer

@registry.register_model
class CapsNet(t2t_model.T2TModel):

    def body(self, features):
        import ipdb;ipdb.set_trace()
        hparams = self._hparams
        targets = features['targets']
        targets = tf.squeeze(targets, 2)
        self.label = targets
        self.X = features['inputs']
        self.build_arch(hparams)

    def build_arch(self, hparams):
        with tf.variable_scope('Conv1_layer'):
            # Conv1, [batch_size, 20, 20, 256]
            conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=256,
                                             kernel_size=9, stride=1,
                                             padding='VALID')
            #assert conv1.get_shape() == [hparams.batch_size, 20, 20, 256]

        # Primary Capsules layer, return [batch_size, 1152, 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(hparams=hparams,num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
            import ipdb;ipdb.set_trace()
            caps1 = primaryCaps(conv1, kernel_size=9, stride=2)
            #assert caps1.get_shape() == [hparams.batch_size, 1152, 8, 1]

        # DigitCaps layer, return [batch_size, 10, 16, 1]
        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(hparams=hparams, num_outputs=10, vec_len=16, with_routing=True, layer_type='FC')
            self.caps2 = digitCaps(caps1)

        # Decoder structure in Fig. 2
        # 1. Do masking, how:
        with tf.variable_scope('Masking'):
            # a). calc ||v_c||, then do softmax(||v_c||)
            # [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2),
                                                  axis=2, keep_dims=True) + hparams.epsilon)
            self.softmax_v = tf.nn.softmax(self.v_length, dim=1)
            #assert self.softmax_v.get_shape() == [hparams.batch_size, 10, 1, 1]

            # b). pick out the index of max softmax val of the 10 caps
            # [batch_size, 10, 1, 1] => [batch_size] (index)
            self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
            #assert self.argmax_idx.get_shape() == [hparams.batch_size, 1, 1]
            self.argmax_idx = tf.reshape(self.argmax_idx, shape=(hparams.batch_size, ))

            # Method 1.
            if not self.hparams.mask_with_y:
                # c). indexing
                # It's not easy to understand the indexing process with argmax_idx
                # as we are 3-dim animal
                masked_v = []
                for batch_size in range(hparams.batch_size):
                    v = self.caps2[batch_size][self.argmax_idx[batch_size], :]
                    masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))

                self.masked_v = tf.concat(masked_v, axis=0)
                #assert self.masked_v.get_shape() == [hparams.batch_size, 1, 16, 1]
            # Method 2. masking with true label, default mode
            else:
                # self.masked_v = tf.matmul(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)), transpose_a=True)
                self.masked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)))
                self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keep_dims=True) + hparams.epsilon)

        # 2. Reconstructe the MNIST images with 3 FC layers
        # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
        with tf.variable_scope('Decoder'):
            vector_j = tf.reshape(self.masked_v, shape=(hparams.batch_size, -1))
            fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
            #assert fc1.get_shape() == [hparams.batch_size, 512]
            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
            #assert fc2.get_shape() == [hparams.batch_size, 1024]
            self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)

    def cap_loss(self, hparams):
        # 1. The margin loss

        # [batch_size, 10, 1, 1]
        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., hparams.m_plus - self.v_length))
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., self.v_length - hparams.m_minus))
        #assert max_l.get_shape() == [hparams.batch_size, 10, 1, 1]

        # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
        max_l = tf.reshape(max_l, shape=(hparams.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(hparams.batch_size, -1))

        # calc T_c: [batch_size, 10]
        # T_c = Y, is my understanding correct? Try it.
        T_c = self.Y
        # [batch_size, 10], element-wise multiply
        L_c = T_c * max_l + hparams.lambda_val * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # 2. The reconstruction loss
        orgin = tf.reshape(self.X, shape=(hparams.batch_size, -1))
        squared = tf.square(self.decoded - orgin)
        self.reconstruction_err = tf.reduce_mean(squared)

        # 3. Total loss
        # The paper uses sum of squared error as reconstruction error, but we
        # have used reduce_mean in `# 2 The reconstruction loss` to calculate
        # mean squared error. In order to keep in line with the paper,the
        # regularization scale should be 0.0005*784=0.392
        self.total_loss = self.margin_loss + hparams.regularization_scale * self.reconstruction_err

    # Summary
    def cap_summary(self,hparams):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/margin_loss', self.margin_loss))
        train_summary.append(tf.summary.scalar('train/reconstruction_loss', self.reconstruction_err))
        train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))
        recon_img = tf.reshape(self.decoded, shape=(hparams.batch_size, 28, 28, 1))
        train_summary.append(tf.summary.image('reconstruction_img', recon_img))
        self.train_summary = tf.summary.merge(train_summary)

        correct_prediction = tf.equal(tf.to_int32(self.labels), self.argmax_idx)
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

@registry.register_hparams
def capsNet_mnist_hparams():
    hparams = common_hparams.basic_params1()
    hparams.epsilon = 1e-9
    hparams.mask_with_y = 1
    hparams.m_plus = 0.9
    hparams.m_minus = 0.1
    hparams.lambda_val = 0.5
    hparams.regularization_scale = 0.392
    hparams.iter_routing = 3
    return hparams