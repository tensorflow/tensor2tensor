class InGraphBatchEnv(object):
  """Abstract class for batch of environments inside the TensorFlow graph.
  """

  def __getattr__(self, name):
    """Forward unimplemented attributes to one of the original environments.

    Args:
      name: Attribute that was accessed.

    Returns:
      Value behind the attribute name in one of the original environments.
    """
    return getattr(self._batch_env, name)

  def __len__(self):
    """Number of combined environments."""
    return len(self._batch_env)

  def __getitem__(self, index):
    """Access an underlying environment by index."""
    return self._batch_env[index]

  def simulate(self, action):
    """Step the batch of environments.

    The results of the step can be accessed from the variables defined below.

    Args:
      action: Tensor holding the batch of actions to apply.

    Returns:
      Operation.
    """
    raise NotImplemented


  def reset(self, indices=None):
    """Reset the batch of environments.

    Args:
      indices: The batch indices of the environments to reset.

    Returns:
      Batch tensor of the new observations.
    """
    raise NotImplemented

  @property
  def observ(self):
    """Access the variable holding the current observation."""
    return self._observ

  def close(self):
    """Send close messages to the external process and join them."""
    self._batch_env.close()
