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
r"""Adversarially attack a model.

This script adversarially attacks a model and evaluates accuracy at various
  epsilons.

Params such as which epsilons to evaluate at and the attack algorithm are
  specified by attack_params, see models/resnet.py for examples.

--ignore_incorrect will only attack those examples that are already correctly
  classified by the model.

--surrogate_attack will attack a model (A) and evaluate adversarial examples for
  A on a different model (B).

Example run:
- train a resnet on cifar10:
    bin/t2t_trainer.py --problem=image_cifar10 --hparams_set=resnet_cifar_32 \
      --model=resnet

- evaluate robustness using the FGSM attack:
    bin/t2t_attack.py --attack_params_set=resnet_fgsm --problem=image_cifar10\
      --hparams_set=resnet_cifar_32 --model=resnet
"""

import os

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import problem as problem_lib  # pylint: disable=unused-import
from tensor2tensor.utils import adv_attack_utils
from tensor2tensor.utils import cloud_mlengine
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

# See flags.py for additional command-line flags.
flags.DEFINE_string("attack_params_set", None,
                    "Which attack parameters to use.")
flags.DEFINE_boolean("surrogate_attack", False,
                     "Perform an attack on a surrogate model.")
flags.DEFINE_string("surrogate_model", None, "Surrogate model to attack.")
flags.DEFINE_string("surrogate_hparams_set", None,
                    "Surrogate model's hyperparameter set.")
flags.DEFINE_string("surrogate_output_dir", None,
                    "Directory storing surrogate model's weights.")
flags.DEFINE_boolean(
    "ignore_incorrect", False, "Ignore examples that are "
    "incorrectly classified to begin with.")


def create_attack_params():
  return registry.attack_params(FLAGS.attack_params_set)


def create_attack(attack):
  return registry.attacks(attack)


def create_surrogate_hparams():
  return trainer_lib.create_hparams(FLAGS.surrogate_hparams_set, None)


def create_surrogate_run_config(hp):
  """Create a run config.

  Args:
    hp: model hyperparameters
  Returns:
    a run config
  """
  save_ckpt_steps = max(FLAGS.iterations_per_loop, FLAGS.local_eval_frequency)
  save_ckpt_secs = FLAGS.save_checkpoints_secs or None
  if save_ckpt_secs:
    save_ckpt_steps = None
  assert FLAGS.surrogate_output_dir
  # the various custom getters we have written do not play well together yet.
  # TODO(noam): ask rsepassi for help here.
  daisy_chain_variables = (
      hp.daisy_chain_variables and hp.activation_dtype == "float32" and
      hp.weight_dtype == "float32")
  return trainer_lib.create_run_config(
      model_dir=os.path.expanduser(FLAGS.surrogate_output_dir),
      master=FLAGS.master,
      iterations_per_loop=FLAGS.iterations_per_loop,
      num_shards=FLAGS.tpu_num_shards,
      log_device_placement=FLAGS.log_device_placement,
      save_checkpoints_steps=save_ckpt_steps,
      save_checkpoints_secs=save_ckpt_secs,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
      num_gpus=FLAGS.worker_gpu,
      gpu_order=FLAGS.gpu_order,
      shard_to_cpu=FLAGS.locally_shard_to_cpu,
      num_async_replicas=FLAGS.worker_replicas,
      gpu_mem_fraction=FLAGS.worker_gpu_memory_fraction,
      enable_graph_rewriter=FLAGS.enable_graph_rewriter,
      use_tpu=FLAGS.use_tpu,
      schedule=FLAGS.schedule,
      no_data_parallelism=hp.no_data_parallelism,
      daisy_chain_variables=daisy_chain_variables,
      ps_replicas=FLAGS.ps_replicas,
      ps_job=FLAGS.ps_job,
      ps_gpu=FLAGS.ps_gpu,
      sync=FLAGS.sync,
      worker_id=FLAGS.worker_id,
      worker_job=FLAGS.worker_job,
      random_seed=FLAGS.random_seed,
      tpu_infeed_sleep_secs=FLAGS.tpu_infeed_sleep_secs,
      inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
      log_step_count_steps=FLAGS.log_step_count_steps,
      intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads)


def prepare_data(problem, hparams, params, config):
  """Construct input pipeline."""
  input_fn = problem.make_estimator_input_fn(
      tf.estimator.ModeKeys.EVAL, hparams, force_repeat=True)
  dataset = input_fn(params, config)
  features, _ = dataset.make_one_shot_iterator().get_next()
  inputs, labels = features["targets"], features["inputs"]
  inputs = tf.to_float(inputs)
  input_shape = inputs.shape.as_list()
  inputs = tf.reshape(inputs, [hparams.batch_size] + input_shape[1:])
  labels = tf.reshape(labels, [hparams.batch_size])
  return inputs, labels, features


def main(argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  trainer_lib.set_random_seed(FLAGS.random_seed)
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
  t2t_trainer.maybe_log_registry_and_exit()


  if FLAGS.cloud_mlengine:
    cloud_mlengine.launch()
    return

  if FLAGS.generate_data:
    t2t_trainer.generate_data()

  if cloud_mlengine.job_dir():
    FLAGS.output_dir = cloud_mlengine.job_dir()

  if argv:
    t2t_trainer.set_hparams_from_args(argv[1:])

  if FLAGS.surrogate_attack:
    tf.logging.warn("Performing surrogate model attack.")
    sur_hparams = create_surrogate_hparams()
    trainer_lib.add_problem_hparams(sur_hparams, FLAGS.problem)

  hparams = t2t_trainer.create_hparams()
  trainer_lib.add_problem_hparams(hparams, FLAGS.problem)

  attack_params = create_attack_params()
  attack_params.add_hparam(attack_params.epsilon_name, 0.0)

  if FLAGS.surrogate_attack:
    sur_config = create_surrogate_run_config(sur_hparams)
  config = t2t_trainer.create_run_config(hparams)
  params = {
      "batch_size": hparams.batch_size,
      "use_tpu": FLAGS.use_tpu,
  }

  # add "_rev" as a hack to avoid image standardization
  problem = registry.problem(FLAGS.problem + "_rev")

  inputs, labels, features = prepare_data(problem, hparams, params, config)

  sess = tf.Session()

  if FLAGS.surrogate_attack:
    sur_model_fn = t2t_model.T2TModel.make_estimator_model_fn(
        FLAGS.surrogate_model, sur_hparams)
    sur_ch_model = adv_attack_utils.T2TAttackModel(
        sur_model_fn, features, params, sur_config, scope="surrogate")
    # Dummy call to construct graph
    sur_ch_model.get_probs(inputs)

    checkpoint_path = os.path.expanduser(FLAGS.surrogate_output_dir)
    tf.contrib.framework.init_from_checkpoint(
        tf.train.latest_checkpoint(checkpoint_path), {"/": "surrogate/"})
    sess.run(tf.global_variables_initializer())

  other_vars = set(tf.global_variables())

  model_fn = t2t_model.T2TModel.make_estimator_model_fn(
      FLAGS.model, hparams)
  ch_model = adv_attack_utils.T2TAttackModel(model_fn, features, params, config)

  acc_mask = None
  probs = ch_model.get_probs(inputs)
  if FLAGS.ignore_incorrect:
    preds = tf.argmax(probs, -1, output_type=labels.dtype)
    preds = tf.reshape(preds, labels.shape)
    acc_mask = tf.to_float(tf.equal(labels, preds))
  one_hot_labels = tf.one_hot(labels, probs.shape[-1])

  if FLAGS.surrogate_attack:
    attack = create_attack(attack_params.attack)(sur_ch_model, sess=sess)
  else:
    attack = create_attack(attack_params.attack)(ch_model, sess=sess)

  new_vars = set(tf.global_variables()) - other_vars

  # Restore weights
  saver = tf.train.Saver(new_vars)
  checkpoint_path = os.path.expanduser(FLAGS.output_dir)
  saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

  # reuse variables
  tf.get_variable_scope().reuse_variables()

  def compute_accuracy(x, l, mask):
    """Compute model accuracy."""
    preds = ch_model.get_probs(x)
    preds = tf.squeeze(preds)
    preds = tf.argmax(preds, -1, output_type=l.dtype)

    _, acc_update_op = tf.metrics.accuracy(l, preds, weights=mask)

    if FLAGS.surrogate_attack:
      preds = sur_ch_model.get_probs(x)
      preds = tf.squeeze(preds)
      preds = tf.argmax(preds, -1, output_type=l.dtype)
      acc_update_op = tf.tuple((acc_update_op,
                                tf.metrics.accuracy(l, preds, weights=mask)[1]))

    sess.run(tf.initialize_local_variables())
    for i in range(FLAGS.eval_steps):
      tf.logging.info(
          "\tEvaluating batch [%d / %d]" % (i + 1, FLAGS.eval_steps))
      acc = sess.run(acc_update_op)
    if FLAGS.surrogate_attack:
      tf.logging.info("\tFinal acc: (%.4f, %.4f)" % (acc[0], acc[1]))
    else:
      tf.logging.info("\tFinal acc: %.4f" % acc)
    return acc

  epsilon_acc_pairs = []
  for epsilon in attack_params.attack_epsilons:
    tf.logging.info("Attacking @ eps=%.4f" % epsilon)
    attack_params.set_hparam(attack_params.epsilon_name, epsilon)
    adv_x = attack.generate(inputs, y=one_hot_labels, **attack_params.values())
    acc = compute_accuracy(adv_x, labels, acc_mask)
    epsilon_acc_pairs.append((epsilon, acc))

  for epsilon, acc in epsilon_acc_pairs:
    if FLAGS.surrogate_attack:
      tf.logging.info(
          "Accuracy @ eps=%.4f: (%.4f, %.4f)" % (epsilon, acc[0], acc[1]))
    else:
      tf.logging.info("Accuracy @ eps=%.4f: %.4f" % (epsilon, acc))


if __name__ == "__main__":
  tf.app.run()
