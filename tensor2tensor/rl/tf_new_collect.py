import tensorflow as tf
import numpy as np
from munch import Munch
tf.enable_eager_execution()

class NewInGraphBatchEnv():

  def __init__(self, batch_size):
    self.batch_size = batch_size

  @property
  def meta_data(self):
    raise NotImplementedError()

  @property
  def empty_hidden_state(self):
    raise NotImplementedError()

  def step(self, hidden_state, action):
    raise NotImplementedError()

  def reset_one_env(self):
    raise NotImplementedError()

  def reset(self, hidden_state, observation, done):
    hidden_state, observation, _ = tf.scan(lambda _, x: tf.cond(x[2],
                                                                lambda: self.reset_one_env() + (tf.constant(False),),
                                                                lambda: x),
                                           (hidden_state, observation, done),
                                           initializer=(hidden_state[0], observation[0], done[0]))
    return hidden_state, observation



class NewSimulatedBatchEnv(NewInGraphBatchEnv):

  def __init__(self, batch_size):
    super(NewSimulatedBatchEnv, self).__init__(batch_size)

  @property
  def meta_data(self):
    return ([self.batch_size, 2, 1], tf.int32, 'hidden_state'), \
           ([self.batch_size, 2, 1], tf.int32, 'observation'), \
           ([self.batch_size], tf.int32, 'action')

  def step(self, hidden_state, action):
    ob = hidden_state[..., -1:] + 1
    new_hidden_state = tf.concat([hidden_state[..., 1:], ob], axis=2)
    done = tf.constant([False, False, False])
    reward = tf.constant([0.1, 0.1, 0.1])

    return new_hidden_state, ob, reward, done

  def reset_one_env(self):
    hidden_state = tf.zeros(self.meta_data[0][0][1:], self.meta_data[0][1]) + 10
    return hidden_state, hidden_state[...,-1:]


class NewStackWrapper(NewInGraphBatchEnv):

  def __init__(self, batch_env, history=4):
    super(NewStackWrapper, self).__init__(batch_env.batch_size)
    self.batch_env = batch_env
    self.history = history

  @property
  def meta_data(self):
    hs, ob, ac = self.batch_env.meta_data
    ob_shape, ob_type, _ = ob
    new_ob_shape = [self.batch_size, self.history] + ob_shape[1:]

    return hs, (new_ob_shape, ob_type, 'observation'), ac




def policy(ob, batch_size):
  action = tf.random.uniform((batch_size,), maxval=9, dtype=tf.int32)
  pdf = tf.random.uniform((batch_size,), dtype=tf.float32)
  value_function = tf.random.uniform((batch_size,), dtype=tf.float32)

  return action, pdf, value_function


batch_env = NewSimulatedBatchEnv(batch_size=3)

def _new_define_collect(batch_env, hparams, force_beginning_resets):

  batch_size = batch_env.batch_size
  hidden_state_type, observation_type, action_type = batch_env.meta_data
  done_type = ([batch_size], tf.bool, 'done')


  ppo_data_metadata =  [observation_type, ([batch_size], tf.float32, 'reward'),
                        done_type, action_type,
                        ([batch_size], tf.float32, 'pdf'),
                        ([batch_size], tf.float32, 'value_function')]

  initial_state_metadata = [hidden_state_type, observation_type, done_type]

  # These are only for typing, values will be discarded
  initial_ppo_batch = tuple(tf.zeros(shape, dtype=type) for shape, type, _ in ppo_data_metadata)

  # Below we intialize with ones, to
  # set done=True. Other fields are just for typeing.
  if force_beginning_resets:
    initial_running_state = tuple(tf.ones(shape, dtype=type)
                                   for shape, type, _ in initial_state_metadata)
  else:
    initial_running_state = tuple(
      tf.get_variable(  # pylint: disable=g-complex-comprehension
        "collect_initial_running_state_%s" % (name),
        shape=shape,
        dtype=dtype,
        initializer=tf.ones_initializer(),
        trainable=False) for (shape, dtype, name) in initial_state_metadata
    )

  initial_batch = initial_running_state + (initial_ppo_batch,)

  def execution_wrapper(hidden_state, observation, done):
    hidden_state, observation = batch_env.reset(hidden_state, observation, done)
    action, pdf, value_function = policy(observation, batch_size)
    hidden_state, new_observation, reward, done = batch_env.step(hidden_state, action)

    return hidden_state, new_observation, done, (observation, reward, done, action, pdf, value_function)

  # This can be replaced with a python for loop
  ret = tf.scan(lambda a, _: execution_wrapper(a[0], a[1], a[2]), tf.range(hparams.epoch_length), initial_batch)

  return ret[3]

x = _new_define_collect(batch_env, Munch(epoch_length=10), True)
print(x[0][:, 0, 0, 0])