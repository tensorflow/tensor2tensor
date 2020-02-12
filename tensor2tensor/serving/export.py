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

"""Export a trained model for serving."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensor2tensor.bin import t2t_trainer
from tensor2tensor.utils import decoding
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir

import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_bool("export_as_tfhub", False,
                     "If True, the model will be exported as tfHub module.")

tf.flags.DEFINE_string(
    "export_dir", None, "Directory, where export model should be stored."
    "If None, the model will be stored in subdirectory "
    "where checkpoints are: --output_dir")

tf.flags.DEFINE_string(
    "checkpoint_path", None, "Which checkpoint to export."
    "If None, we will use the latest checkpoint stored in the directory "
    "specified by --output_dir")

tf.flags.DEFINE_bool(
    "as_text", True,
    "Whether to write the SavedModel proto in text format. Defaults to `False`."
)


def _get_hparams_path():
  """Get hyper-parameters file path."""
  hparams_path = None
  if FLAGS.output_dir:
    hparams_path = os.path.join(FLAGS.output_dir, "hparams.json")
  elif FLAGS.checkpoint_path:  # Infer hparams.json from checkpoint path
    hparams_path = os.path.join(
        os.path.dirname(FLAGS.checkpoint_path), "hparams.json")

  # Check if hparams_path really exists
  if hparams_path:
    if tf.gfile.Exists(hparams_path):
      tf.logging.info("hparams file %s exists", hparams_path)
    else:
      tf.logging.info("hparams file %s does not exist", hparams_path)
      hparams_path = None

  # Can't find hparams_path
  if not hparams_path:
    tf.logging.warning(
        "--output_dir not specified or file hparams.json does not exists. "
        "Hyper-parameters will be infered from --hparams_set and "
        "--hparams only. These may not match training time hyper-parameters.")

  return hparams_path


def create_estimator(run_config, hparams):
  return trainer_lib.create_estimator(
      FLAGS.model,
      hparams,
      run_config,
      decode_hparams=decoding.decode_hparams(FLAGS.decode_hparams),
      use_tpu=FLAGS.use_tpu,
      export_saved_model_api_version=FLAGS.export_saved_model_api_version,
      use_guarantee_const_getter=FLAGS.use_guarantee_const_getter)


def create_hparams():
  """Create hyper-parameters object."""
  return trainer_lib.create_hparams(
      FLAGS.hparams_set,
      FLAGS.hparams,
      data_dir=os.path.expanduser(FLAGS.data_dir),
      problem_name=FLAGS.problem,
      hparams_path=_get_hparams_path())


# TODO(michalski): Move this method into tfhub utils.
def export_module_spec_with_checkpoint(module_spec,
                                       checkpoint_path,
                                       export_path,
                                       scope_prefix=""):
  """Exports given checkpoint as tfhub module with given spec."""

  # The main requirement is that it is possible to know how to map from
  # module variable name to checkpoint variable name.
  # This is trivial if the original code used variable scopes,
  # but can be messy if the variables to export are interwined
  # with variables not export.
  with tf.Graph().as_default():
    m = hub.Module(module_spec)
    assign_map = {
        scope_prefix + name: value for name, value in m.variable_map.items()
    }
    tf.train.init_from_checkpoint(checkpoint_path, assign_map)
    init_op = tf.initializers.global_variables()
    with tf.Session() as session:
      session.run(init_op)
      m.export(export_path, session)


def export_as_tfhub_module(model_name,
                           hparams,
                           decode_hparams,
                           problem,
                           checkpoint_path,
                           export_dir):
  """Exports the last checkpoint from the directory as tfhub module.

  It creates the Module spec and signature (based on T2T problem information),
  which is later used to create and export the hub module.
  Module will be saved inside the ckpt_dir.

  Args:
    model_name: name of the model to be exported.
    hparams: T2T parameters, model graph will be based on them.
    decode_hparams: T2T parameters for decoding.
    problem: the name of the problem
    checkpoint_path: path to the checkpoint to be exported.
    export_dir: Directory to write the exported model to.
  """

  def hub_module_fn():
    """Creates the TF graph for the hub module."""
    model_fn = t2t_model.T2TModel.make_estimator_model_fn(
        model_name,
        hparams,
        decode_hparams=decode_hparams,
        use_tpu=FLAGS.use_tpu)
    features = problem.serving_input_fn(
        hparams, decode_hparams, use_tpu=FLAGS.use_tpu).features

    # we must do a copy of the features, as the model_fn can add additional
    # entries there (like hyperparameter settings etc).
    original_features = features.copy()
    spec = model_fn(features, labels=None, mode=tf.estimator.ModeKeys.PREDICT)

    hub.add_signature(
        inputs=original_features,
        outputs=spec.export_outputs["serving_default"].outputs)

  # TFHub doesn't support the following collections.
  drop_collections = [tf.GraphKeys.LOSSES,
                      tf.GraphKeys.SUMMARIES, tf.GraphKeys.LOCAL_VARIABLES]
  module_spec = hub.create_module_spec(
      hub_module_fn, drop_collections=drop_collections)
  # Loads the weights from the checkpoint using the model above
  # and saves it in the export_path.
  export_module_spec_with_checkpoint(
      module_spec,
      checkpoint_path=checkpoint_path,
      export_path=export_dir,
      scope_prefix="")


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  trainer_lib.set_random_seed(FLAGS.random_seed)
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)

  if FLAGS.checkpoint_path:
    checkpoint_path = FLAGS.checkpoint_path
    ckpt_dir = os.path.dirname(checkpoint_path)
  else:
    ckpt_dir = os.path.expanduser(FLAGS.output_dir)
    checkpoint_path = tf.train.latest_checkpoint(ckpt_dir)

  hparams = create_hparams()
  hparams.no_data_parallelism = True  # To clear the devices
  problem = hparams.problem
  decode_hparams = decoding.decode_hparams(FLAGS.decode_hparams)

  export_dir = FLAGS.export_dir or os.path.join(ckpt_dir, "export")

  if FLAGS.export_as_tfhub:
    checkpoint_path = tf.train.latest_checkpoint(ckpt_dir)
    export_as_tfhub_module(FLAGS.model, hparams, decode_hparams, problem,
                           checkpoint_path, export_dir)
    return

  run_config = t2t_trainer.create_run_config(hparams)

  estimator = create_estimator(run_config, hparams)

  exporter = tf.estimator.FinalExporter(
      "exporter",
      lambda: problem.serving_input_fn(hparams, decode_hparams, FLAGS.use_tpu),
      as_text=FLAGS.as_text)

  exporter.export(
      estimator,
      export_dir,
      checkpoint_path=checkpoint_path,
      eval_result=None,
      is_the_final_export=True)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
