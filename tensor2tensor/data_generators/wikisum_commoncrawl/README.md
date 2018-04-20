# Generating Wikipedia by Summarizing Long Sequences

This directory contains the code and scripts to generate the dataset from the
paper [Generating Wikipedia by Summarizing Long
Sequences](https://arxiv.org/abs/1801.10198).  The task is to generate a
Wikipedia article based on the contents of the cited references in that article
and the top 10 Google search results for the article's title.  The references
are extracted from [CommonCrawl](http://commoncrawl.org/), an open-source crawl
of the web.

## Support files

**URLs:** The dataset contains ~90M URLs total (~2.3M Wikipedia articles, each
with ~40 reference URLs). The URLs in the dataset are available in sharded JSON
files here: `gs://tensor2tensor-data/wikisum/wiki_urls/`.

**Wikipedia Articles:** We have processed the Wikipedia articles slightly to
extract the title, section breaks, and section headings. The processed Wikipedia
content is available in sharded `TFRecord` files containing serialized
`tensorflow.Example` protocol buffers here:
`gs://tensor2tensor-data/wikisum/wiki_content/`. The sharding is determined by a
hash of the Wikpedia article's title. The `Example`s contain features `[url,
title, section_titles, section_texts]`.

**CommonCrawl References Index:** To enable efficiently extracting the reference
URLs from CommonCrawl, we provide a JSON file per CommonCrawl file which maps a
reference URL contained in that CommonCrawl file to a list of shard ids:
`gs://tensor2tensor-data/wikisum/commoncrawl_metadata/`. These shards are the
ones that contain one or more Wikipedia articles that cite this reference. The
scripts in this directory will use this information to efficiently join the
reference with their Wikipedia articles.

*Note*: You can use [`gsutil`](https://cloud.google.com/storage/docs/gsutil) to
view the support files.

## Data generation

We provide instructions to generate the data on a single machine or on a cluster
of machines. Because of the size of the dataset and the size of the CommonCrawl
data, single-machine data generation is quite slow and it is highly recommended
to parallelize across machines.

Data generation will extract the references from CommonCrawl, generate a
vocabulary, join the references with their Wikipedia articles, run TF-IDF to
rank reference paragraphs for a given article, and then encode the references
and the Wikipedia article with the vocabulary and write the encoded training or
evaluation example out to disk.

The output of data generation is a set of `TFRecord` files containing serialized
`tensorflow.Example` protocol buffers, with feature keys `"inputs"` and
`"targets"`. The inputs are the reference tokens, and the targets are the
Wikipedia article tokens.

### Multiple machines

Using multiple machines, data generation has 3 steps:

1. Extract references
1. Build vocabulary
1. Produce Examples

`extract_references.py` and `produce_examples.py` both have `--task_id` and
`--num_tasks` command-line flags; `--num_tasks` is the total number of tasks
running in parallel and should be the same across all tasks, and `--task_id`
should be set to the id of the current task `[0, num_tasks)`.

With 1,000 tasks and a good internet connection to AWS S3 buckets (where the
CommonCrawl data is hosted), data generation takes well under 24 hours.

Extract references:

```
python -m \
  tensor2tensor.data_generators.wikisum_commoncrawl.extract_references \
  --num_tasks=$NUM_TASKS \
  --task_id=$TASK_ID \
  --out_dir=$TMP_DIR
```

Generate vocabulary:

```
python -m \
  tensor2tensor.data_generators.wikisum_commoncrawl.generate_vocab \
  --out_dir=$DATA_DIR \
  --refs_dir=$TMP_DIR
```

Produce output examples:

```
python -m \
  tensor2tensor.data_generators.wikisum_commoncrawl.produce_examples \
  --num_tasks=$NUM_TASKS \
  --task_id=$TASK_ID \
  --out_dir=$DATA_DIR \
  --refs_dir=$TMP_DIR \
  --vocab_dir=$DATA_DIR
```

### Single machine

To generate data on a single machine:

```
t2t-datagen --problem=wikisum_commoncrawl --data_dir=$DATA_DIR --tmp_dir=$TMP_DIR
```

This will use Python's `multiprocessing` module to use multiple cores, but it is
still quite slow, and depending on your internet connection speed, could take
days/weeks.

See
`tensor2tensor.data_generators.wikisum_commoncrawl.wikisum_commoncrawl.WikisumCommoncrawl.generate_data`
for the
implementation.

## Training

```
t2t-trainer \
  --problem=wikisum_commoncrawl \
  --model=transformer \
  --hparams_set=transformer_base \
  --train_steps=250000 \
  --eval_steps=100 \
  --data_dir=$DATA_DIR \
  --output_dir=$TRAIN_DIR
```
