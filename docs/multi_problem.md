# Multi-problem training

Multi-problem training is possible by defining [MultiProblem](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/multi_problem.py) sub-classes that specify a list of [Problem](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/problem.py) objects to include in training. In some cases, multi-problem training can be used to improve performance compared to training on individual problems.

In the following sections we'll discuss MultiProblem from a usage perspective followed by that of someone wishing to build upon it.

Please note the [T2T Walkthrough](https://github.com/tensorflow/tensor2tensor/blob/master/docs/walkthrough.md) documentation is a good place to start to understand the variety of component concepts we'll build on here.

## Usage

### Problem definition and datagen

In this discussion we'll consider the following (large) multi-problem that includes ten different sub-problems. These include:

1. A [language modeling](https://en.wikipedia.org/wiki/Language_model) [problem](https://github.com/tensorflow/tensor2tensor/blob/0dff89d64c3406d42717280cb9135a5ce7af793c/tensor2tensor/data_generators/wiki_lm.py#L223) operating on a corpus of German, English, French, and Romanian language wikipedia articles.
2. Multiple compatible pairwise lanugage translation problems (En -> De, En -> Fr, En -> Ro, De -> En, Fr -> En, Ro -> En)
3. A compatible [version](https://github.com/tensorflow/tensor2tensor/blob/ef12bee72270b322165d073c39a650a189de39aa/tensor2tensor/data_generators/cnn_dailymail.py#L267) of the combined CNN/DailyMail news article summarization problem.
4. A compatible [version](https://github.com/tensorflow/tensor2tensor/blob/ef12bee72270b322165d073c39a650a189de39aa/tensor2tensor/data_generators/multinli.py#L155) of the [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/) textual entailment classification problem.
5. A compatible [version](https://github.com/tensorflow/tensor2tensor/blob/1de13dbebccb415d89b0658e18a57e9607bafd32/tensor2tensor/data_generators/squad.py#L126) of the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) question/answer problem.

```python

@registry.register_problem
class LanguagemodelMultiWikiTranslate(multi_problem.MultiProblem):
  """Wiki multi-lingual LM and multiple translations."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(LanguagemodelMultiWikiTranslate, self).__init__(
        was_reversed, was_copy)
    self.task_list.append(wiki_lm.LanguagemodelDeEnFrRoWiki64k())
    self.task_list.append(translate_ende.TranslateEndeWmtMulti64k())
    self.task_list.append(translate_enfr.TranslateEnfrWmtMulti64k())
    self.task_list.append(translate_enro.TranslateEnroWmtMultiTiny64k())
    self.task_list.append(translate_ende.TranslateEndeWmtMulti64k(
        was_reversed=True))
    self.task_list.append(translate_enfr.TranslateEnfrWmtMulti64k(
        was_reversed=True))
    self.task_list.append(translate_enro.TranslateEnroWmtMultiTiny64k(
        was_reversed=True))
    self.task_list.append(
        cnn_dailymail.SummarizeCnnDailymailWikiLMMultiVocab64k())
    self.task_list.append(multinli.MultiNLIWikiLMMultiVocab64k())
    self.task_list.append(squad.SquadConcatMulti64k())

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD

```

The word "compatible" was used a lot above! That's because each of these problems have been modified to use the vocabulary produced by the Wikipedia-based language modeling problem, e.g. the following

```python
@registry.register_problem
class SummarizeCnnDailymailWikiLMMultiVocab64k(SummarizeCnnDailymail32k):
  """Summarize CNN and Daily Mail articles using multi-lingual 64k vocab."""

  @property
  def vocab_filename(self):
    return wiki_lm.LanguagemodelDeEnFrRoWiki64k().vocab_filename
```

**Important note:** It's easy to miss the key point that, as implemented currently, the first task in the task list must be a language modelling problem and each included task must be modified to use the resulting vocabulary.

With a propperly defined and registered multi-problem we can now run datagen as follows:

```bash

t2t_datagen --problem=languagemodel_multi_wiki_translate \
    --model=transformer \
    --data_dir=/tmp/mydatadir \
    --tmp_dir=/tmp/mytmpdir \
    --output_dir ~/t2t_train/transformer_multi_2jan19

```

This will generate examples into the provided `data_dir`.

### Training

Next we're ready to try training a model on this MultiProblem.

```bash

t2t_trainer --problem=languagemodel_multi_wiki_translate \
    --model=transformer \
    --data_dir=/tmp/mydatadir \
    --hparams_set=transformer_tall_pretrain_lm_tpu_adafactor_large \
    --output_dir ~/t2t_train/transformer_multi_2jan19

```

The `hparams_set` parameter we provided above was [transformer_tall_pretrain_lm_tpu_adafactor_large](https://github.com/tensorflow/tensor2tensor/blob/08e83030acf3ef13d15ad6eaefaa0a67fb20b59d/tensor2tensor/models/transformer.py#L1721), also provided below:

```python

@registry.register_hparams
def transformer_tall_pretrain_lm_tpu_adafactor_large():
  """Hparams for transformer on LM pretraining on TPU, large model."""
  hparams = transformer_tall_pretrain_lm_tpu_adafactor()
  hparams.hidden_size = 1024
  hparams.num_heads = 16
  hparams.filter_size = 32768  # max fitting in 16G memory is 49152, batch 2
  hparams.batch_size = 4
  hparams.multiproblem_mixing_schedule = "constant"
  # Task order: lm/en-de/en-fr/en-ro/de-en/fr-en/ro-en/cnndm/mnli/squad.
  hparams.multiproblem_per_task_threshold = "320,80,160,2,80,160,2,20,5,5"
  return hparams

```

Here it's worth noting a couple things, one that we have specified a `multi_problem_mixing_schedule` (which is required), consumed by [MultiProblem.mix_data](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/multi_problem.py#L280). More on this below but in short here we're just sampling training examples from each problem with equal probability and without making this a function of the training step.

Here we have also specified a (non-required) `multiproblem_per_task_threshold` parameter, also consumed by mix_data, and specifically used by [sample_task](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/multi_problem.py#L340) to inform a weighted random sampling - i.e. with weights 1 and 9, sampling uniformly, the first would be sampled 1/10 of the time and the other 9/10.

### Inference

You can try translating from English to German using a model previously trained on `LanguagemodelMultiWikiTranslate` (the one shown above) ([gs://tensor2tensor-checkpoints/transformer_multi_2jan19/](https://console.cloud.google.com/storage/browser/tensor2tensor-checkpoints/transformer_multi_2jan19/)). Just copy the checkpoint down to a local directory such as the one given via `--output_dir` below:

```bash

t2t_decoder --problem=languagemodel_multi_wiki_translate \
    --model=transformer \
    --hparams_set=transformer_tall_pretrain_lm_tpu_adafactor_large \
    --decode_hparams='batch_size=1,multiproblem_task_id=64510' \
    --hparams="" \
    --output_dir ~/t2t_train/transformer_multi_2jan19 \
    --decode_from_file ~/newstest2014.en

```

The file passed to `--decode_from_file` is simply a file with one sentence to translate on each line (in its original form, not post-vocabulary-encoded).

A key requirement for multi-problem inference is that we specify the ID of the problem for which we want to perform inference. But wait, why is the task ID 64510? We can see from the code for [`MultiProblem.update_task_ids`](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/multi_problem.py#L386) that TID's have a place at the end of the vocabulary.

```python

class MultiProblem(problem.Problem):
  """MultiProblem base class."""

  ...

  def update_task_ids(self, encoder_vocab_size):
    """Generate task_ids for each problem.
    These ids correspond to the index of the task in the task_list.
    Args:
      encoder_vocab_size: the size of the vocab which is used to compute
        the index offset.
    """
    for idx, task in enumerate(self.task_list):
      task.set_task_id(idx + encoder_vocab_size)
      tf.logging.info("Task %d (%s) has id %d." %
                      (idx, task.name, task.task_id))

```

And further by examining the code for [`MultiProblem.get_hparams`](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/multi_problem.py#L160), shown below, we can see the vocab size considered above is the base 64k vocab used for the language modelling problem plus the maximum number of classes across any problem included in the multi-problem that is a classification problem (thereby overloading these class label fields across problems):

```python

class MultiProblem(problem.Problem):
  """MultiProblem base class."""

  ...

  def get_hparams(self, model_hparams=None):
    if self._hparams is not None:
      return self._hparams
    self._hparams = self.task_list[0].get_hparams(model_hparams)
    # Increase the vocab size to account for task ids and modify the modality.
    vocab_size_inc = len(self.task_list)
    vocab_size_inc += self.get_max_num_classes()
    vocab_size = self._hparams.vocabulary["targets"].vocab_size
    new_vocab_size = vocab_size + vocab_size_inc
    if model_hparams.multiproblem_vocab_size > new_vocab_size:
      new_vocab_size = model_hparams.multiproblem_vocab_size
    tf.logging.info("Old vocabulary size: %d" % vocab_size)
    self.update_task_ids(vocab_size)
    tf.logging.info("New vocabulary size: %d" % new_vocab_size)
    self._hparams.vocab_size["targets"] = new_vocab_size
    self._hparams.modality["targets"] = modalities.SymbolModality(
        model_hparams, self._hparams.vocab_size["targets"])
    return self._hparams

```

**TODO: The math doesn't check out here for me. It looks like the max class size is 3, the base vocab size should be 64k, and we're doing inference with the second problem in the set, so by this math the ID should be 64005. Perhaps the vocab here is ~64k and was optimized to be 64505 - 64510 - 2 (the second problem) - 3 (MultiNLI has three classes)?**
