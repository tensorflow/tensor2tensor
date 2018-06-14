# Generating Wikipedia by Summarizing Long Sequences

This directory contains the code and scripts to generate the dataset from the
paper [Generating Wikipedia by Summarizing Long
Sequences](https://arxiv.org/abs/1801.10198).  The task is to generate a
Wikipedia article based on the contents of the cited references in that article
and the top 10 Google search results for the article's title.

There are 2 sources for the reference URLs used:

1. [CommonCrawl](http://commoncrawl.org/), an open-source crawl of the web. The
   advantage of using CommonCrawl is that the dataset is perfectly reproducible.
   However, there is limited coverage of the reference URLs.
1. Live web fetches. Coverage is considerably increased, but the content is
   subject to change.

This document provides instructions for producing both datasets.

## Support files

Some files that are used in dataset generation have already been generated and
uploaded to Google Cloud Storage as `gs://tensor2tensor-data/wikisum`.

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

Data generation will first extract reference content (from either CommonCrawl or
the web), then generate a vocabulary, join the references with their Wikipedia
articles, run TF-IDF to rank reference paragraphs for a given article, and then
encode the references and the Wikipedia article with the vocabulary and write
the encoded training or evaluation example out to disk.

The output of data generation is a set of `TFRecord` files containing serialized
`tensorflow.Example` protocol buffers, with feature keys `"inputs"` and
`"targets"`. The inputs are the reference tokens, and the targets are the
Wikipedia article tokens.

In both cases, you must use multiple machines to extract references and produce
the final data to disk because of the size of the data. See `parallel_launch.py`
which is a script that will launch N machines in parallel on GCP. You can use it
as a guide if you'd like to launch on other infrastructure.

There are 3 jobs to run:

1. Extract references: `get_references_commoncrawl.py` for `WikisumCommoncrawl`
   and `get_references_web.py` for `WikisumWeb`.
1. Build vocabulary (single-machine): `generate_vocab.py`
1. Produce Examples: `produce_examples.py`

With 1,000 machines with a good internet connection, data generation takes well
under 24 hours.

## Setup if using `parallel_launch.py` to launch on Google Cloud Platform

First, [install the `gcloud` CLI](https://cloud.google.com/sdk/downloads).

```
# Initialize the CLI
gcloud init

# Login
gcloud auth login

# Update the CLI
gcloud components update

# Set the default project and zone
gcloud config set core/project myproject
gcloud config set compute/zone us-central1-c
```

You'll also need to request the requisite
[quotas](https://console.cloud.google.com/iam-admin/quotas) in the zone you'll
be launching the machines in (whatever default zone you set above):

* In-use IP addresses: 1,000
* Internal IP addresses: 1,000
* Persistent Disk Standard (GB): 10,000
* CPUs: 4,000

**Running the commands below will launch instances on Google Cloud Platform and
you will incur charges.** If any of the commands go bad, immediately delete any
stranded instances. `delete_instances.sh` helps you delete instances in bulk
from the command-line, or you can delete many instances at once from the
[GCP Console](https://console.cloud.google.com/).

### Cost estimates

These are rough (and **not** guaranteed) estimates of cost if you were to launch
on GCP.

Pricing is taken from
[here](https://cloud.google.com/compute/pricing#custommachinetypepricing).

* `WikisumCommoncrawl`
  * `get_references_commoncrawl`: $50 (1k machines, 1 CPU, 2G memory, 1 hour)
  * `produce_examples`: $25 (1k machines, 1 CPU, 3G memory, 30 minutes)
* `WikisumWeb`
  * `get_references_web`: $600 (1k machines, 4 CPU, 4G memory, 4 hours)
  * `produce_examples`: $25 (1k machines, 1 CPU, 3G memory, 30 minutes)

## Commands to generate `WikisumCommoncrawl`

```
pip install tensor2tensor -U --user

# Set to your own GCS bucket
BUCKET=gs://my-gcs-bucket/wikisum_commoncrawl

# Extract references from CommonCrawl
python -m tensor2tensor.data_generators.wikisum.parallel_launch \
  --num_instances=1000 \
  --cpu=1 --mem=2 \
  --name=wikisum-cc-refs \
  --log_dir=$BUCKET/logs \
  --setup_command="pip install tensor2tensor tensorflow -U -q --user" \
  --command_prefix="python -m tensor2tensor.data_generators.wikisum.get_references_commoncrawl --num_tasks=1000 --out_dir=$BUCKET/wiki_references --task_id"

# Generate vocabulary file
python -m tensor2tensor.data_generators.wikisum.generate_vocab \
  --out_dir=$BUCKET/data \
  --refs_dir=$BUCKET/wiki_references \
  --for_commoncrawl

# Produce examples
python -m tensor2tensor.data_generators.wikisum.parallel_launch \
  --num_instances=1000 \
  --cpu=1 --mem=3 \
  --name=wikisum-cc-produce \
  --log_dir=$BUCKET/logs \
  --setup_command="pip install tensor2tensor tensorflow -U -q --user" \
  --command_prefix="python -m tensor2tensor.data_generators.wikisum.produce_examples --out_dir=$BUCKET/data --refs_dir=$BUCKET/wiki_references --num_tasks=1000 --for_commoncrawl --task_id"

# Validate data
python -m tensor2tensor.data_generators.wikisum.validate_data \
  --out_dir=$BUCKET/data \
  --for_commoncrawl
```

## Commands to generate `WikisumWeb`

```
pip install tensor2tensor -U --user

# Set to your own GCS bucket
BUCKET=gs://my-gcs-bucket/wikisum_web

# Fetch references from web
python -m tensor2tensor.data_generators.wikisum.parallel_launch \
  --num_instances=1000 \
  --cpu=4 --mem=4 \
  --name=wikisum-web-refs \
  --log_dir=$BUCKET/logs \
  --setup_command="pip3 install tensorflow tensor2tensor aiohttp cchardet aiodns bs4 -U -q --user" \
  --command_prefix="python3 wikisum/get_references_web.py --out_dir=$BUCKET/wiki_references --shard_id"

# Generate vocabulary file
python -m tensor2tensor.data_generators.wikisum.generate_vocab \
  --out_dir=$BUCKET/data \
  --refs_dir=$BUCKET/wiki_references

# Produce examples
python -m tensor2tensor.data_generators.wikisum.parallel_launch \
  --num_instances=1000 \
  --cpu=1 --mem=3 \
  --name=wikisum-web-produce \
  --log_dir=$BUCKET/logs \
  --setup_command="pip install tensor2tensor tensorflow -U -q --user" \
  --command_prefix="python -m tensor2tensor.data_generators.wikisum.produce_examples --out_dir=$BUCKET/data --refs_dir=$BUCKET/wiki_references --num_tasks=1000 --task_id"

# Validate data
python -m tensor2tensor.data_generators.wikisum.validate_data \
  --out_dir=$BUCKET/data
```

## Training

**TODO(rsepassi)**: Put actual results achieved on `wikisum_web` and/or
`wikisum_commoncrawl` and with what `hparams_set`.

```
PROBLEM=wikisum_web  # or wikisum_commoncrawl
t2t-trainer \
  --problem=$PROBLEM \
  --model=transformer \
  --hparams_set=transformer_base \
  --train_steps=250000 \
  --eval_steps=100 \
  --data_dir=$DATA_DIR \
  --output_dir=$TRAIN_DIR
```
