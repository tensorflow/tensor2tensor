# This is the initial version of atari wrappers written in t2t. We assume that wrappers take as input
# a class of the interface bach env (todo: pm, fill me)
import tensorflow as tf


from tensor2tensor.rl.envs.in_graph_batch_env import InGraphBatchEnv


class WrapperBase(InGraphBatchEnv):

  def __init__(self, batch_env):
    self._length = len(batch_env)
    self._batch_env = batch_env
    self.action_shape = batch_env.action_shape
    self.action_dtype = batch_env.action_dtype

  @property
  def observ(self):
    """Access the variable holding the current observation."""
    return self._observ

  def __len__(self):
    """Number of combined environments."""
    return self._length

  def reset(self, indices=None):
    return self._batch_env.reset(indices)


class TransformWrapper(WrapperBase):

  def __init__(self, batch_env, transform_observation=None,
               transform_reward=tf.identity, transform_done=tf.identity):
    super().__init__(batch_env)
    if transform_observation:
      _, observ_shape, observ_dtype = transform_observation
      self._observ = tf.Variable(
          tf.zeros((self.length,) + observ_shape, observ_dtype),trainable=False)
    else:
      self._observ = self._batch_env.observ

    self.transform_observation = transform_observation
    self.transform_reward = transform_reward
    self.transform_done = transform_done

  def simulate(self, action):
    with tf.name_scope('environment/simulate'): #TODO: Do we need this?
      reward, done = self._batch_env.simulate(action)
      with tf.control_dependencies([reward]):
        if self.transform_observation:
          observ = self.transform_observation[0](self._batch_env.observ)
          assign_op = self._observ.assign(observ)
        else:
          assign_op =tf.no_op()
        with tf.control_dependencies([assign_op]):
          return self.transform_reward(reward), self.transform_done(done)


class WarpFrameWrapper(TransformWrapper):

  def __init__(self, batch_env):
    """Warp frames to 84x84 as done in the Nature paper and later work."""

    width, height = 84, 84
    nature_transform = \
      lambda o: tf.image.rgb_to_grayscale(tf.image.resize_images(o, [width, height]))

    super().__init__(batch_env, transform_observation=nature_transform)


class PongT2TGeneratorHackWrapper(TransformWrapper):

  def __init__(self, batch_env, add_value):
    shift_reward = lambda r: tf.add(r, add_value)

    super().__init__(batch_env, transform_reward=shift_reward)


class MaxAndSkipWrapper(WrapperBase):

  def __init__(self, batch_env, skip=4):
    super().__init__(batch_env)
    self.skip = skip
    self._observ = None
    observs_shape = batch_env.observ.shape
    observ_dtype = tf.float32

    self._observ = tf.Variable(tf.zeros(observs_shape, observ_dtype), trainable=False)

  def simulate(self, action):
    with tf.name_scope('environment/simulate'): #TODO: Do we need this?

      initializer = (tf.zeros_like(self._observ), tf.fill((len(self),), 0.0), tf.fill((len(self),), False))

      def not_done_step(a, _):
        reward, done = self._batch_env.simulate(action)
        with tf.control_dependencies([reward, done]):
          r0 = tf.maximum(a[0], self._batch_env.observ)
          r1 = tf.add(a[1], reward)
          r2 = tf.logical_or(a[2], done)

          return (r0, r1, r2)

      simulate_ret = tf.scan(not_done_step, tf.range(self.skip), initializer=initializer, parallel_iterations=1, infer_shape=False)
      simulate_ret = [ret[-1, ...] for ret in simulate_ret]

      with tf.control_dependencies([self._observ.assign(simulate_ret[0])]):
        return tf.identity(simulate_ret[1]), tf.identity(simulate_ret[2])


class MomoryWrapper(WrapperBase):

  #This is a singleton class
  singleton = None

  def __init__(self, batch_env):
    super().__init__(batch_env)
    assert MomoryWrapper.singleton==None, "The class cannot be instatiated multiple times"
    MomoryWrapper.singleton = self
    assert self._length==1, "We support only one environment"

    observ_dtype = tf.float32

    infinity = 10000000
    self._speculum = tf.FIFOQueue(infinity, dtypes=[observ_dtype, tf.float32, tf.int32, tf.bool])

    self._observ = self._batch_env.observ


  def simulate(self, action):
    with tf.name_scope('environment/simulate'): #TODO: Do we need this?
      reward, done = self._batch_env.simulate(action)
      with tf.control_dependencies([reward, done]):

        enqueue_op = self._speculum.enqueue([self._batch_env.observ, reward, action, done])

        with tf.control_dependencies([enqueue_op]):
          return tf.identity(reward), tf.identity(done)