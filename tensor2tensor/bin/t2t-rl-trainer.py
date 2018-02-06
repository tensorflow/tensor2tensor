#!/usr/bin/env python
"""t2t-rl-trainer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.bin import t2t_rl_trainer

import tensorflow as tf

def main(argv):
  t2t_rl_trainer.main(argv)


if __name__ == "__main__":
  tf.app.run()
