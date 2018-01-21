#encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np

from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.models.capsule.capsNet import CapsNet, capsNet_mnist_hparams
from tensor2tensor.utils import registry

import tensorflow as tf

class capsule_mnist_test(tf.test.TestCase):

    def _testCapsule(self, img_size, output_size):
        vocab_size = 10
        batch_size = 3
        import ipdb;ipdb.set_trace()
        x = np.random.random_integers(0, high=1, size = (batch_size, img_size, img_size, 1))
        y = np.random.random_integers(1, high=vocab_size, size=(batch_size,1,1,1))
        hparams = capsNet_mnist_hparams()
        p_hparams = problem_hparams.test_problem_hparams(vocab_size, vocab_size)
        p_hparams.input_modality['inputs'] = (registry.Modalities.IMAGE, None)
        with self.test_session() as session:
            features = {
                "inputs": tf.constant(x,dtype=tf.int32),
                "targets": tf.constant(y, dtype=tf.int32)
            }
            model = CapsNet(hparams, tf.estimator.ModeKeys.TRAIN, p_hparams)
            logits, _ = model(features)
            session.run(tf.global_variables_initializer())
            res = session.run(logits)

    def testCapsuleSmall(self):
        import ipdb;ipdb.set_trace()
        self._testCapsule(img_size=9, output_size=None)

if __name__ == '__main__':
    tf.test.main()