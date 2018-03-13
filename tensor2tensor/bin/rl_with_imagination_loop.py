import os
import tempfile

from tensor2tensor import problems
from tensor2tensor.bin import t2t_trainer
from tensor2tensor.utils import trainer_lib
from tensor2tensor.rl import rl_trainer_lib

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


def main(_):
  # Setup data directories
  prefix = "~/trash/loop_0309"
  data_dir = os.path.expanduser(prefix + "/data")
  tmp_dir = os.path.expanduser(prefix + "/tmp")
  output_dir = os.path.expanduser(prefix + "/output")
  tf.gfile.MakeDirs(data_dir)
  tf.gfile.MakeDirs(tmp_dir)
  tf.gfile.MakeDirs(output_dir)
  last_model = "/home/blazej.osinski/t2t_rl_data_0309/data/ppo_Xth32m/model30.ckpt"
  for iloop in range(10):
      # 1. generate data from policy
      # if iloop == 0:
      #     gym_problem = problems.problem("gym_pong_random5k")
      # else:

      print("  >>>   Step {}.1. - generate data from policy".format(iloop))
      FLAGS.problems = "gym_pong_trajectories_from_policy_base"
      FLAGS.model_path = last_model
      gym_problem = problems.problem(FLAGS.problems)
      iter_data_dir = os.path.join(data_dir, str(iloop))
      tf.gfile.MakeDirs(iter_data_dir)
      gym_problem.generate_data(iter_data_dir, tmp_dir)

      print("  >>> Step {}.2. - generate env model".format(iloop))
      # 2. generate env model
      FLAGS.data_dir = iter_data_dir
      FLAGS.output_dir = output_dir
      FLAGS.model = "basic_conv_gen"
      FLAGS.hparams_set = "basic_conv_small"
      FLAGS.train_steps = 100
      FLAGS.eval_steps = 10
      t2t_trainer.main([])

      # 3. train PPO in model env
      print("  >>> Step {}.3. - train PPO in model env".format(iloop))
      iteration_num=3
      hparams = trainer_lib.create_hparams("atari_base", "epochs_num={},simulated_environment=True,eval_every_epochs=0,save_models_every_epochs={}".format(iteration_num+1, iteration_num))
      ppo_dir = tempfile.mkdtemp(dir=data_dir, prefix="ppo_")
      rl_trainer_lib.train(hparams, "PongNoFrameskip-v4", ppo_dir)

      last_model =  ppo_dir + "/model{}.ckpt".format(iteration_num)

if __name__ == "__main__":
  tf.app.run()
