import os
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import text_problems


@registry.register_model
class SimilarityTransformer(t2t_model.T2TModel):
  # pylint: disable=abstract-method

  """
  This class defines the model to compute similarity scores between functions and
  docstrings
  """

  def __init__(self, *args, **kwargs):
    super(SimilarityTransformer, self).__init__(*args, **kwargs)


  def body(self, features):
    # TODO: need to fill this with Transformer encoder/decoder
    # and loss calculation
    raise NotImplementedError


@registry.register_problem
class GithubFunctionDocstring(text_problems.Text2TextProblem):
  # pylint: disable=abstract-method

  """This class defines the problem of finding similarity between Python function
   and docstring"""

  @property
  def is_generate_per_split(self):
    return False

  def generate_samples(self, data_dir, _tmp_dir, dataset_split):  #pylint: disable=no-self-use
    """This method returns the generator to return {"inputs": [text], "targets": [text]} dict"""

    functions_file_path = os.path.join(data_dir, '{}.function'.format(dataset_split))
    docstrings_file_path = os.path.join(data_dir, '{}.docstring'.format(dataset_split))

    return text_problems.text2text_txt_iterator(functions_file_path, docstrings_file_path)
