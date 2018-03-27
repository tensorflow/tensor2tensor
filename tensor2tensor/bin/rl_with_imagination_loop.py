import os
import tempfile
from tensor2tensor import problems
from tensor2tensor.bin import t2t_trainer
from tensor2tensor.utils import trainer_lib
from tensor2tensor.rl import rl_trainer_lib
from tensor2tensor.rl.envs.tf_atari_wrappers import PongT2TGeneratorHackWrapper
from tensor2tensor.rl.envs.tf_atari_wrappers import TimeLimitWrapper
import tensorflow as tf
import time
import datetime


flags = tf.flags
FLAGS = flags.FLAGS


def train(hparams, output_dir):
  prefix = output_dir
  #remove trash
  # prefix = "~/trash/loop_{}".format(random.randint(10000, 99999))
  data_dir = os.path.expanduser(prefix + "/data")
  tmp_dir = os.path.expanduser(prefix + "/tmp")
  output_dir = os.path.expanduser(prefix + "/output")
  tf.gfile.MakeDirs(data_dir)
  tf.gfile.MakeDirs(tmp_dir)
  tf.gfile.MakeDirs(output_dir)
  last_model = ""
  start_time = time.time()
  line = ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    "
  for iloop in range(hparams.epochs):
      time_delta = time.time() - start_time
      print(line+"Step {}.1. - generate data from policy. "
            "Time: {}".format(iloop, str(datetime.timedelta(seconds=time_delta))))
      FLAGS.problems = "gym_discrete_problem"
      FLAGS.agent_policy_path = last_model
      gym_problem = problems.problem(FLAGS.problems)
      gym_problem.num_steps = hparams.true_env_generator_num_steps
      iter_data_dir = os.path.join(data_dir, str(iloop))
      tf.gfile.MakeDirs(iter_data_dir)
      gym_problem.generate_data(iter_data_dir, tmp_dir)

      time_delta = time.time() - start_time
      print(line+"Step {}.2. - generate env model. "
            "Time: {}".format(iloop, str(datetime.timedelta(seconds=time_delta))))
      # 2. generate env model
      FLAGS.data_dir = iter_data_dir
      FLAGS.output_dir = output_dir
      FLAGS.model = "basic_conv_gen"
      FLAGS.hparams_set = "basic_conv_small"
      FLAGS.train_steps = hparams.model_train_steps
      FLAGS.eval_steps = 1
      t2t_trainer.main([])

      time_delta = time.time() - start_time
      print(line+"Step {}.3. - evalue env model. "
            "Time: {}".format(iloop, str(datetime.timedelta(seconds=time_delta))))
      gym_simulated_problem = problems.problem("gym_simulated_discrete_problem")
      gym_simulated_problem.num_steps = hparams.simulated_env_generator_num_steps
      gym_simulated_problem.generate_data(iter_data_dir, tmp_dir)

      time_delta = time.time() - start_time
      print(line+"Step {}.4. - train PPO in model env."
            " Time: {}".format(iloop, str(datetime.timedelta(seconds=time_delta))))
      ppo_epochs_num=hparams.ppo_epochs_num
      ppo_hparams = trainer_lib.create_hparams("atari_base", "epochs_num={},simulated_environment=True,eval_every_epochs=0,save_models_every_epochs={}".format(ppo_epochs_num+1, ppo_epochs_num),
                                           data_dir=output_dir)
      ppo_hparams.epoch_length = hparams.ppo_epoch_length
      ppo_dir = tempfile.mkdtemp(dir=data_dir, prefix="ppo_")
      in_graph_wrappers = [(TimeLimitWrapper, {"timelimit": 150}),
                           (PongT2TGeneratorHackWrapper, {"add_value": -2})] + gym_problem.in_graph_wrappers
      ppo_hparams.add_hparam("in_graph_wrappers", in_graph_wrappers)
      rl_trainer_lib.train(ppo_hparams, "PongNoFrameskip-v4", ppo_dir)

      last_model = ppo_dir + "/model{}.ckpt".format(ppo_epochs_num)


def main(_):
    train(1)


if __name__ == "__main__":
  tf.app.run()