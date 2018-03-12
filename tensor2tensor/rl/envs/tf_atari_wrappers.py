# This is the initial version of atari wrappers written in t2t. We assume that wrappers take as input
# a class of the interface bach env (todo: pm, fill me)
import tensorflow as tf

class WarpFrame():

  def __init__(self, batch_env):
    """Warp frames to 84x84 as done in the Nature paper and later work."""
    self.width = 84
    self.height = 84
    self.length = batch_env.length
    self.wrapped_env = batch_env

    self.action_shape = batch_env.action_shape
    self.action_dtype = batch_env.action_dtype
    observ_shape = (self.width, self.height, 1)
    observ_dtype = tf.float32

    with tf.variable_scope('env_temporary'):
      self._observ = tf.Variable(
          tf.zeros((self.length,) + observ_shape, observ_dtype),
          name='observ', trainable=False)


  def simulate(self, action):

    with tf.name_scope('environment/simulate'): #TODO: Do we need this?
      reward, done = self.wrapped_env.simulate(action)
      with tf.control_dependencies([reward]):
        #This is the key point, all the rest is the tf bullshit
        observ = tf.image.resize_images(self.wrapped_env.observ, [self.width, self.height])
        observ = tf.image.rgb_to_grayscale(observ)
        observ = tf.Print(observ, [observ], "w frame = ")
        #End of important code. The bullshit restarts
        with tf.control_dependencies([self._observ.assign(observ)]):
          return tf.identity(reward), tf.identity(done)

  def reset(self, indices=None):
    return self.wrapped_env.reset()


  #TODO: possibly move it superclass
  @property
  def observ(self):
    """Access the variable holding the current observation."""
    return self._observ

  def __len__(self):
    """Number of combined environments."""
    return self.length


class MaxAndSkipEnv():

  def __init__(self, batch_env, skip=4):
    self.length = batch_env.length
    self.wrapped_env = batch_env
    self.skip = 4

    self.action_shape = batch_env.action_shape
    self.action_dtype = batch_env.action_dtype
    observs_shape = batch_env.observ.shape
    observ_dtype = tf.float32

    with tf.variable_scope('env_temporary'):
      self._observ = tf.Variable(
          tf.zeros(observs_shape, observ_dtype),
          name='observ', trainable=False)




  def simulate(self, action):

    with tf.name_scope('environment/simulate'): #TODO: Do we need this?

      initializer = (tf.zeros_like(self._observ), tf.fill((self.length,), 0.0), tf.fill((self.length,), False))

      def not_done_step(a, x):
        reward, done = self.wrapped_env.simulate(action)
        with tf.control_dependencies([reward, done]):
          r0 = tf.maximum(a[0], self.wrapped_env.observ)
          r1 = tf.add(a[1], reward)
          r2 = tf.logical_or(a[2], done)

          return (r0, r1, r2)


      simulate_ret = tf.scan(not_done_step, tf.range(self.skip), initializer=initializer)
      simulate_ret = [ret[-1, ...] for ret in simulate_ret]

      simulate_ret[0] = tf.Print(simulate_ret[0], [simulate_ret[0]], "w max = ")

      with tf.control_dependencies([self._observ.assign(simulate_ret[0])]):
        return tf.identity(simulate_ret[1]), tf.identity(simulate_ret[2])

  def reset(self, indices=None):
    return self.wrapped_env.reset()


  #TODO: possibly move it superclass
  @property
  def observ(self):
    """Access the variable holding the current observation."""
    return self._observ

  def __len__(self):
    """Number of combined environments."""
    return self.length
