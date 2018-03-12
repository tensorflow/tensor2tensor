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

"""Translate a file with all checkpoints in a given directory.

t2t-decoder will be executed with these parameters:
--problems
--data_dir
--output_dir with the value of --model_dir
--decode_from_file with the value of --source
--decode_hparams with properly formatted --beam_size and --alpha
--checkpoint_path automatically filled
--decode_to_file automatically filled
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

# Dependency imports

from tensor2tensor.utils import bleu_hook

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

# t2t-translate-all specific options
flags.DEFINE_string("decoder_command", "t2t-decoder {params}",
                    "Which command to execute instead t2t-decoder. "
                    "{params} is replaced by the parameters. Useful e.g. for "
                    "qsub wrapper.")
flags.DEFINE_string("model_dir", "",
                    "Directory to load model checkpoints from.")
flags.DEFINE_string("source", None,
                    "Path to the source-language file to be translated")
flags.DEFINE_string("translations_dir", "translations",
                    "Where to store the translated files.")
flags.DEFINE_integer("min_steps", 0, "Ignore checkpoints with less steps.")
flags.DEFINE_integer("wait_minutes", 0,
                     "Wait upto N minutes for a new checkpoint")

# options derived from t2t-decoder
flags.DEFINE_integer("beam_size", 4, "Beam-search width.")
flags.DEFINE_float("alpha", 0.6, "Beam-search alpha.")
flags.DEFINE_string("model", "transformer", "see t2t-decoder")
flags.DEFINE_string("t2t_usr_dir", None, "see t2t-decoder")
flags.DEFINE_string("data_dir", None, "see t2t-decoder")
flags.DEFINE_string("problems", None, "see t2t-decoder")
flags.DEFINE_string("hparams_set", "transformer_big_single_gpu",
                    "see t2t-decoder")


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  # pylint: disable=unused-variable
  model_dir = os.path.expanduser(FLAGS.model_dir)
  translations_dir = os.path.expanduser(FLAGS.translations_dir)
  source = os.path.expanduser(FLAGS.source)
  tf.gfile.MakeDirs(translations_dir)
  translated_base_file = os.path.join(translations_dir, FLAGS.problems)

  # Copy flags.txt with the original time, so t2t-bleu can report correct
  # relative time.
  flags_path = os.path.join(translations_dir, FLAGS.problems + "-flags.txt")
  if not os.path.exists(flags_path):
    shutil.copy2(os.path.join(model_dir, "flags.txt"), flags_path)

  locals_and_flags = {"FLAGS": FLAGS}
  for model in bleu_hook.stepfiles_iterator(model_dir, FLAGS.wait_minutes,
                                            FLAGS.min_steps):
    tf.logging.info("Translating " + model.filename)
    out_file = translated_base_file + "-" + str(model.steps)
    locals_and_flags.update(locals())
    if os.path.exists(out_file):
      tf.logging.info(out_file + " already exists, so skipping it.")
    else:
      tf.logging.info("Translating " + out_file)
      params = (
          "--t2t_usr_dir={FLAGS.t2t_usr_dir} --output_dir={model_dir} "
          "--data_dir={FLAGS.data_dir} --problems={FLAGS.problems} "
          "--decode_hparams=beam_size={FLAGS.beam_size},alpha={FLAGS.alpha} "
          "--model={FLAGS.model} --hparams_set={FLAGS.hparams_set} "
          "--checkpoint_path={model.filename} --decode_from_file={source} "
          "--decode_to_file={out_file}"
      ).format(**locals_and_flags)
      command = FLAGS.decoder_command.format(**locals())
      tf.logging.info("Running:\n" + command)
      os.system(command)
  # pylint: enable=unused-variable


if __name__ == "__main__":
  tf.app.run()
