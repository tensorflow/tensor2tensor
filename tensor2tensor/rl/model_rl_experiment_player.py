# This is a starter for piotrmilos experiments. Should not be committed into repo
# The intention is to keep my dirty configs outside of the repo.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.rl.model_rl_experiment import create_loop_hparams, train, rl_modelrl_base
from tensor2tensor.utils import registry

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

@registry.register_hparams
def pm_rl_modelrl_tiny():
  """Tiny set for testing."""
  tiny_hp = tf.contrib.training.HParams(
      epochs=2,
      true_env_generator_num_steps=20,
      model_train_steps=10,
      simulated_env_generator_num_steps=20,
      ppo_epochs_num=2,
      ppo_time_limit=20,
      ppo_epoch_length=20,

  )
  return rl_modelrl_base().override_from_dict(tiny_hp.values())

@registry.register_hparams
def pm_rl_modelrl_tiny_2agents():
  """Tiny set for testing."""
  tiny_hp = tf.contrib.training.HParams(
      epochs=2,
      true_env_generator_num_steps=200,
      model_train_steps=2,
      simulated_env_generator_num_steps=20,
      ppo_epochs_num=2,
      ppo_time_limit=20,
      ppo_epoch_length=20,
      ppo_num_agents=2

  )
  return rl_modelrl_base().override_from_dict(tiny_hp.values())


@registry.register_hparams
def pm_rl_modelrl_longpong_tiny():
  """Tiny set for testing."""
  tiny_hp = tf.contrib.training.HParams(
      epochs=2,
      true_env_generator_num_steps=20,
      model_train_steps=10,
      simulated_env_generator_num_steps=20,
      ppo_epochs_num=2,
      ppo_time_limit=20,
      #The same as GymWrappedLongPongRandom.num_testing_steps
      #both should be roughly similar
      ppo_epoch_length=100,
      game="wrapped_long_pong",

  )
  return rl_modelrl_base().override_from_dict(tiny_hp.values())


@registry.register_hparams
def pm_rl_modelrl_medium():
  """Tiny set for testing."""
  tiny_hp = tf.contrib.training.HParams(
      epochs=2,
      true_env_generator_num_steps=50000,
      model_train_steps=15000,
      simulated_env_generator_num_steps=10000,
      ppo_epochs_num=2,
      ppo_time_limit=20,
      ppo_epoch_length=20,
  )
  return rl_modelrl_base().override_from_dict(tiny_hp.values())

def main(_):
  hp = create_loop_hparams()
  output_dir = FLAGS.output_dir
  train(hp, output_dir)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
