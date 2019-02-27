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




class NewSimulatedBatchEnv(NewInGraphBatchEnv):

  def __init__(self, batch_size):
    super(NewSimulatedBatchEnv, self).__init__(batch_size)

  @property
  def meta_data(self):
    return [([self.batch_size, 2, 1], tf.int32, 'hidden_state')], \
           ([self.batch_size, 2, 1], tf.int32, 'observation'), \
           ([self.batch_size], tf.int32, 'action')

  def step(self, hidden_state, action):
    hidden_state_unpacked = hidden_state[0]
    ob = hidden_state_unpacked[..., -1:] + 1
    new_hidden_state = tf.concat([hidden_state_unpacked[..., 1:], ob], axis=2)
    done = tf.constant([False, False, False])
    reward = tf.constant([0.1, 0.1, 0.1])

    return (new_hidden_state,), ob, reward, done

  def reset(self, hidden_state, observation, done):
    hidden_state_unpacked = hidden_state[0]
    new_hidden_state, observation, _ = tf.scan(lambda _, x: tf.cond(x[2],
                                                                lambda: self._reset_one_env() + (tf.constant(False),),
                                                                lambda: x),
                                           (hidden_state_unpacked, observation, done),
                                           initializer=(hidden_state_unpacked[0], observation[0], done[0]))
    return (new_hidden_state, ), observation


  def _reset_one_env(self):
    hidden_state_single_env = tf.zeros(self.meta_data[0][0][0][1:], self.meta_data[0][0][1]) + 10
    return hidden_state_single_env, hidden_state_single_env[...,-1:]


class NewStackWrapper(NewInGraphBatchEnv):
  #TODO: it might be more natural to stack on the last dim

  def __init__(self, batch_env, history=4):
    super(NewStackWrapper, self).__init__(batch_env.batch_size)
    self._env = batch_env
    self.history = history

  @property
  def meta_data(self):
    hs, ob, ac = self._env.meta_data
    ob_shape, ob_type, _ = ob
    stack_ob_shape = [self.batch_size, self.history] + ob_shape[1:]
    stack_ob_spec = (stack_ob_shape, ob_type, 'observation')

    return hs+[stack_ob_spec], stack_ob_spec, ac

  def step(self, hidden_state, action):
    env_hidden_state, stack_hidden_state = hidden_state
    (new_env_hidden_state,), env_ob, env_reward, env_done = self._env.step((env_hidden_state,), action)
    new_stack_hidden_state = tf.concat([stack_hidden_state[:, 1:, ...], tf.expand_dims(env_ob, axis=1)], axis=1)

    return (new_env_hidden_state, new_stack_hidden_state), new_stack_hidden_state, env_reward, env_done

  def reset(self, hidden_state, _, done):
    env_hidden_state, stack_hidden_state = hidden_state
    env_observ = stack_hidden_state[:, -1, ...]
    (new_env_hidden_state,), new_env_observation = self._env.reset((env_hidden_state,), env_observ, done)

    def extend(ob):
      _, ob_metadata, _ = self._env.meta_data
      multiples = (self.history,) + (1,) * (len(ob_metadata) - 1)
      return tf.tile(tf.expand_dims(ob, axis=0), multiples)

    new_stack_hidden_state, _, _ = tf.scan(lambda _, x: tf.cond(x[2],
                                                                lambda: (extend(x[1]), x[1], tf.constant(False)),
                                                                lambda: x),
                                           (stack_hidden_state, new_env_observation, done),
                                           initializer=(stack_hidden_state[0], new_env_observation[0], done[0]))



    return (new_env_hidden_state, new_stack_hidden_state), new_stack_hidden_state


def policy(ob, batch_size):
  action = tf.random.uniform((batch_size,), maxval=9, dtype=tf.int32)
  pdf = tf.random.uniform((batch_size,), dtype=tf.float32)
  value_function = tf.random.uniform((batch_size,), dtype=tf.float32)

  return action, pdf, value_function


batch_env = NewSimulatedBatchEnv(batch_size=3)
batch_env = NewStackWrapper(batch_env, history=4)

def _new_define_collect(batch_env, hparams, force_beginning_resets):

  batch_size = batch_env.batch_size
  hidden_state_types, observation_type, action_type = batch_env.meta_data
  done_type = ([batch_size], tf.bool, 'done')

  ppo_data_metadata = [observation_type, ([batch_size], tf.float32, 'reward'),
                       done_type, action_type,
                       ([batch_size], tf.float32, 'pdf'),
                       ([batch_size], tf.float32, 'value_function')]

  initial_state_metadata = hidden_state_types + [ observation_type, done_type]

  # These are only for typing, values will be discarded
  initial_ppo_batch = tuple(tf.zeros(shape, dtype=type) for shape, type, _ in ppo_data_metadata)

  # Below we intialize with ones, to
  # set done=True. Other fields are just for typeing.
  if force_beginning_resets:
    initial_running_state = [tf.ones(shape, dtype=type)
                                   for shape, type, _ in initial_state_metadata]
  else:
    initial_running_state = [
      tf.get_variable(  # pylint: disable=g-complex-comprehension
        "collect_initial_running_state_%s" % (name),
        shape=shape,
        dtype=dtype,
        initializer=tf.ones_initializer(),
        trainable=False) for (shape, dtype, name) in initial_state_metadata
    ]

  initial_running_state = tf.contrib.framework.nest.pack_sequence_as(
    ((1,) * len(hidden_state_types),) + (2, 3), initial_running_state)


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
print(x[0][:, 0, :, 0, 0])