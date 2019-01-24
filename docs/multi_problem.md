# Multi-problem training

Multi-problem training is possible by defining [MultiProblem](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/multi_problem.py) sub-classes that specify a list of [Problem](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/problem.py) objects to include in training. In some cases, multi-problem training can be used to improve performance compared to training on individual problems.

In the following sections we'll discuss MultiProblem from a usage perspective followed by that of someone wishing to build upon it.

Please note the [T2T Walkthrough](https://github.com/tensorflow/tensor2tensor/blob/master/docs/walkthrough.md) documentation is a good place to start to understand the variety of component concepts we'll build on here.

## Usage

### Problem definition and datagen

In this discussion we'll consider the following (large) multi-problem that includes ten different sub-problems. These include:

1. A [language modeling](https://en.wikipedia.org/wiki/Language_model) [problem](https://github.com/tensorflow/tensor2tensor/blob/0dff89d64c3406d42717280cb9135a5ce7af793c/tensor2tensor/data_generators/wiki_lm.py#L223) operating on a corpus of German, English, French, and Romanian language wikipedia articles.
2. Multiple compatible pairwise language translation problems (En -> De, En -> Fr, En -> Ro, De -> En, Fr -> En, Ro -> En)
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

With a properly defined and registered multi-problem we can now run datagen as follows:

```bash

t2t-datagen --problem=languagemodel_multi_wiki_translate 

```

This will take approximately the following amount of space (and several hours):

```bash
(t2t) username@instance-2:~$ du -sh /tmp
99G     /tmp
(t2t) username@instance-2:~$ du -sh /tmp/t2t_datagen
81G     /tmp/t2t_datagen
```

### Training

Next we're ready to try training a model on this MultiProblem. Note that by not specifying `--data_dir` above TFExample's were by default generated into /tmp so that's what we'll explicitly provide here.

```bash

t2t-trainer --problem=languagemodel_multi_wiki_translate \
    --model=transformer \
    --hparams_set=transformer_tall_pretrain_lm_tpu_adafactor_large \
    --output_dir ~/t2t_train/transformer_multi_2jan19 \
    --data_dir=/tmp \
    --train_steps=1 \
    --eval_steps=1

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

Here it's worth noting a couple things, one that we have specified a `multi_problem_mixing_schedule` (which is required), consumed by [MultiProblem.mix_data](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/multi_problem.py#L280). When set to "constant" the strategy for sampling examples is not a function of step and is proportional only to the per-task "thresholds" which are by default equal (sample examples from each problem with equal probability).

But notice we have also specified the (non-required) `multiproblem_per_task_threshold` parameter, also consumed by mix_data, and specifically used by [sample_task](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/multi_problem.py#L340) which defines non-uniform thresholds to inform a weighted random sampling. E.g. for two problems with weights 1 and 9 the first would be sampled 1/10 of the time and the other 9/10.

### Inference

You can try translating from English to German using a model previously trained on `LanguagemodelMultiWikiTranslate` (the one shown above) ([gs://tensor2tensor-checkpoints/transformer_multi_2jan19/](https://console.cloud.google.com/storage/browser/tensor2tensor-checkpoints/transformer_multi_2jan19/)). Just copy the checkpoint down to a local directory such as the one given via `--output_dir` below:

```bash

t2t-decoder --problem=languagemodel_multi_wiki_translate \
    --model=transformer \
    --hparams_set=transformer_tall_pretrain_lm_tpu_adafactor_large \
    --decode_hparams='batch_size=1,multiproblem_task_id=64510' \
    --hparams="" \
    --output_dir=~/t2t_train/transformer_multi_2jan19 \
    --decode_from_file ~/newstest2014.en \
    --data_dir=~/t2t_train/transformer_multi_2jan19

```

Here we'll point `--data_dir` to the checkpoint directory which includes the vocab file `vocab.languagemodel_de_en_fr_ro_wiki64k.64000.subwords`; typically data_dir would point to the directory containing your TFRecord example dataset(s).

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

We can look up the task_id that is assigned to each task we may want to use for inference by instantiating the MultiProblem subclass and obtaining the value, in this case via the following:

```python

task_index = 1 # The second task in the list is En -> De
LanguagemodelMultiWikiTranslate().task_list[task_index].task_id

```

For me running the `t2t-decode` command provided above gave the following output:

```bash
...

INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Inference results INPUT: hello world was the news of the day
INFO:tensorflow:Inference results OUTPUT: Hallo Welt war die Nachricht des Tages
INFO:tensorflow:Elapsed Time: 37.15079
INFO:tensorflow:Averaged Single Token Generation Time: 3.3009222 (time 36.3101439 count 11)

...

```