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

"""Evaluate BLEU score for all checkpoints/translations in a given directory.

This script can be used in two ways.


To evaluate one already translated file:

```
t2t-bleu --translation=my-wmt13.de --reference=wmt13_deen.de
```

To evaluate all translations in a given directory (translated by
`t2t-translate-all`):

```
t2t-bleu
  --translations_dir=my-translations
  --reference=wmt13_deen.de
  --event_dir=events
```

In addition to the above-mentioned required parameters,
there are optional parameters:
 * bleu_variant: cased (case-sensitive), uncased, both (default).
 * tag_suffix: Default="", so the tags will be BLEU_cased and BLEU_uncased.
   tag_suffix can be used e.g. for different beam sizes if these should be
   plotted in different graphs.
 * min_steps: Don't evaluate checkpoints with less steps.
   Default=-1 means check the `last_evaluated_step.txt` file, which contains
   the number of steps of the last successfully evaluated checkpoint.
 * report_zero: Store BLEU=0 and guess its time based on the oldest file in the
   translations_dir. Default=True. This is useful, so TensorBoard reports
   correct relative time for the remaining checkpoints. This flag is set to
   False if min_steps is > 0.
 * wait_minutes: Wait upto N minutes for a new translated file. Default=0.
   This is useful for continuous evaluation of a running training, in which case
   this should be equal to save_checkpoints_secs/60 plus time needed for
   translation plus some reserve.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from tensor2tensor.utils import bleu_hook
import tensorflow.compat.v1 as tf


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("source", None,
                    "Path to the source-language file to be translated")
flags.DEFINE_string("reference", None, "Path to the reference translation file")
flags.DEFINE_string("translation", None,
                    "Path to the MT system translation file")
flags.DEFINE_string("translations_dir", None,
                    "Directory with translated files to be evaluated.")
flags.DEFINE_string("event_dir", None, "Where to store the event file.")

flags.DEFINE_string("bleu_variant", "both",
                    "Possible values: cased(case-sensitive), uncased, "
                    "both(default).")
flags.DEFINE_string("tag_suffix", "",
                    "What to add to BLEU_cased and BLEU_uncased tags.")
flags.DEFINE_integer("min_steps", -1,
                     "Don't evaluate checkpoints with less steps.")
flags.DEFINE_integer("wait_minutes", 0,
                     "Wait upto N minutes for a new checkpoint, cf. "
                     "save_checkpoints_secs.")
flags.DEFINE_bool("report_zero", None,
                  "Store BLEU=0 and guess its time based on the oldest file.")


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  if FLAGS.translation:
    if FLAGS.translations_dir:
      raise ValueError(
          "Cannot specify both --translation and --translations_dir.")
    if FLAGS.bleu_variant in ("uncased", "both"):
      bleu = 100 * bleu_hook.bleu_wrapper(FLAGS.reference, FLAGS.translation,
                                          case_sensitive=False)
      print("BLEU_uncased = %6.2f" % bleu)
    if FLAGS.bleu_variant in ("cased", "both"):
      bleu = 100 * bleu_hook.bleu_wrapper(FLAGS.reference, FLAGS.translation,
                                          case_sensitive=True)
      print("BLEU_cased = %6.2f" % bleu)
    return

  if not FLAGS.translations_dir:
    raise ValueError(
        "Either --translation or --translations_dir must be specified.")
  transl_dir = os.path.expanduser(FLAGS.translations_dir)
  if not os.path.exists(transl_dir):
    exit_time = time.time() + FLAGS.wait_minutes * 60
    tf.logging.info("Translation dir %s does not exist, waiting till %s.",
                    transl_dir, time.asctime(time.localtime(exit_time)))
    while not os.path.exists(transl_dir):
      time.sleep(10)
      if time.time() > exit_time:
        raise ValueError("Translation dir %s does not exist" % transl_dir)

  last_step_file = os.path.join(FLAGS.event_dir, "last_evaluated_step.txt")
  if FLAGS.min_steps == -1:
    if tf.gfile.Exists(last_step_file):
      with open(last_step_file) as ls_file:
        FLAGS.min_steps = int(ls_file.read())
    else:
      FLAGS.min_steps = 0
  if FLAGS.report_zero is None:
    FLAGS.report_zero = FLAGS.min_steps == 0

  writer = tf.summary.FileWriter(FLAGS.event_dir)
  for transl_file in bleu_hook.stepfiles_iterator(
      transl_dir, FLAGS.wait_minutes, FLAGS.min_steps, path_suffix=""):
    # report_zero handling must be inside the for-loop,
    # so we are sure the transl_dir is already created.
    if FLAGS.report_zero:
      all_files = (os.path.join(transl_dir, f) for f in os.listdir(transl_dir))
      start_time = min(
          os.path.getmtime(f) for f in all_files if os.path.isfile(f))
      values = []
      if FLAGS.bleu_variant in ("uncased", "both"):
        values.append(tf.Summary.Value(
            tag="BLEU_uncased" + FLAGS.tag_suffix, simple_value=0))
      if FLAGS.bleu_variant in ("cased", "both"):
        values.append(tf.Summary.Value(
            tag="BLEU_cased" + FLAGS.tag_suffix, simple_value=0))
      writer.add_event(tf.summary.Event(summary=tf.Summary(value=values),
                                        wall_time=start_time, step=0))
      FLAGS.report_zero = False

    filename = transl_file.filename
    tf.logging.info("Evaluating " + filename)
    values = []
    if FLAGS.bleu_variant in ("uncased", "both"):
      bleu = 100 * bleu_hook.bleu_wrapper(FLAGS.reference, filename,
                                          case_sensitive=False)
      values.append(tf.Summary.Value(tag="BLEU_uncased" + FLAGS.tag_suffix,
                                     simple_value=bleu))
      tf.logging.info("%s: BLEU_uncased = %6.2f" % (filename, bleu))
    if FLAGS.bleu_variant in ("cased", "both"):
      bleu = 100 * bleu_hook.bleu_wrapper(FLAGS.reference, filename,
                                          case_sensitive=True)
      values.append(tf.Summary.Value(tag="BLEU_cased" + FLAGS.tag_suffix,
                                     simple_value=bleu))
      tf.logging.info("%s: BLEU_cased = %6.2f" % (transl_file.filename, bleu))
    writer.add_event(tf.summary.Event(
        summary=tf.Summary(value=values),
        wall_time=transl_file.mtime, step=transl_file.steps))
    writer.flush()
    with open(last_step_file, "w") as ls_file:
      ls_file.write(str(transl_file.steps) + "\n")


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
