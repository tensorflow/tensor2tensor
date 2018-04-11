# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data generator and model for translation
with multiple source features."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import os

# Dependency imports

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities

import tensorflow as tf


FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID


class SourceFeatureProblem(translate.TranslateProblem):
  """Problem spec for En-Fr translation with source features."""
    
  @property
  def approx_vocab_size(self):
    raise NotImplementedError()

  @property
  def vocab_filename(self):
    raise NotImplementedError()

  @property
  def use_subword_tags(self):
    r"""use subword tags"""
    raise NotImplementedError()
    
  @property
  def sfeat_delimiter(self):
    r"""Source feature delimiter in feature file"""
    raise NotImplementedError()

  def vocab_sfeat_filenames(self, f_id):
    r"""One vocab per feature type"""
    raise NotImplementedError()

  def vocab_data_files(self):
    raise NotImplementedError()
  
  def source_data_files(self, dataset_split):
    raise NotImplementedError()
  
  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    sample_iterator = super().generate_samples(data_dir, tmp_dir, dataset_split)
    
    datasets = self.source_data_files(dataset_split)
    tag = "train" if dataset_split == problem.DatasetSplit.TRAIN else "dev"
    data_path = self.compile_sfeat_data(tmp_dir, datasets, "%s-compiled-%s" % (self.name, tag))
        
    # create source feature vocabularies
    self.get_or_create_src_feature_vocabs(data_dir, tmp_dir)
    
    sfeat_iterator = text_problems.txt_line_iterator(data_path + ".sfeat")
    for sample in sample_iterator:
      sample["sfeats"] = next(sfeat_iterator)
      yield sample

  def get_or_create_src_feature_vocabs(self, data_dir, tmp_dir):
    r"""
    Generate as many vocabularies as there are source feature types.
    """
    source = self.vocab_data_files()[0][2]
    vocab_file = os.path.join(data_dir, self.vocab_sfeat_filenames(0))
    if os.path.isfile(vocab_file):
      tf.logging.info("Found source feature vocabs: %s", vocab_file[:-1]+"*")
      return

    filepath = os.path.join(tmp_dir, source)
    sfeat_vocab_lists = defaultdict(lambda: set())
    tf.logging.info("Generating source feature vocabs from %s", filepath)
    with tf.gfile.GFile(filepath, mode="r") as source_file:
      for line in source_file:
        feat_sets = [fs.split(self.sfeat_delimiter) for fs in line.strip().split()]
        for f_id, _ in enumerate(feat_sets[0]):
          feat = [fs[f_id] for fs in feat_sets]
          sfeat_vocab_lists[f_id].update(feat)
                    
    if self.use_subword_tags:
      tf.logging.info("Generating subword tag vocab")
      f_id = len(sfeat_vocab_lists)
      sfeat_vocab_lists[f_id] = {"B", "I", "E", "O"}

    sfeat_vocabs = {}
    for f_id in sfeat_vocab_lists:
      vocab = text_encoder.TokenTextEncoder(
        vocab_filename=None,
        vocab_list=sfeat_vocab_lists[f_id])  
      vocab_filepath = os.path.join(data_dir, self.vocab_sfeat_filenames(f_id))
      vocab.store_to_file(vocab_filepath)

  def compile_sfeat_data(self, tmp_dir, datasets, filename):
    filename = os.path.join(tmp_dir, filename)
    src_feat_fname = filename + '.sfeat'
    for dataset in datasets:
      try:
        src_feat_filepath = os.path.join(tmp_dir, dataset[2])
      except IndexError:
        if self.use_subword_tags:
          raise IndexError("No source feature file given.",
                           "Using only subword tags is not allowed." )
        else:
          raise IndexError("No source feature file given.")
      with tf.gfile.GFile(src_feat_fname, mode="w") as sf_resfile:
        with tf.gfile.Open(src_feat_filepath) as f:
          for src_feats in f:
            sf_resfile.write(src_feats.strip())
            sf_resfile.write("\n")
    return filename

  def get_subword_tags(self, subword_nb):
    r"""Get subword tags as an additional feature:
    B: beginning of a word
    I: inside
    E: end
    O: full word
    """
    if subword_nb == 1:
      feat = ['O']
    else:
      feat = ['B', 'E']
      while len(feat) < subword_nb:
        feat.insert(1, 'I')
    return feat
            
  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    txt_encoder = self.get_or_create_vocab(data_dir, tmp_dir)
    feat_encoders = self.get_src_feature_encoders(data_dir, tmp_dir)

    for sample in generator:
      new_input = []
      split_counter = []
      for token in sample["inputs"].split():
        if self.vocab_type == "subwords":
          new_toks = txt_encoder.encode_without_tokenizing(token)
        elif self.vocab_type == "tokens":
          new_toks = txt_encoder.encode(token)
        else:
          raise ValueError("VocabType not supported")
        new_input += new_toks
        split_counter.append(len(new_toks))
      sample["inputs"] = new_input
      sample["inputs"].append(EOS)

      feat_seqs = defaultdict(lambda: [])
      for tok_id, feat_set in enumerate(sample["sfeats"].split()):
        for f_id, feat in enumerate(feat_set.split(self.sfeat_delimiter)):
          # synchronize feature with subword
          feat = [feat] * split_counter[tok_id]
          feat_seqs[f_id] += feat
        if self.use_subword_tags:
          f_id += 1
          feat = self.get_subword_tags(split_counter[tok_id])
          feat_seqs[f_id] += feat
 
      del sample["sfeats"]

      for f_id in range(len(feat_seqs)):
        fs = feat_encoders[f_id].encode(' '.join(feat_seqs[f_id]))
        fs.append(EOS)
        assert len(fs) == len(sample["inputs"]), "Source word and feature sequences must have the same length"
        sample["sfeats.%d" %f_id] = fs

      if self.vocab_type == "subwords":
        sample["targets"] = txt_encoder.encode_without_tokenizing(sample["targets"])
      elif self.vocab_type == "tokens":
        sample["targets"] = txt_encoder.encode(sample["targets"])
      else:
        raise ValueError("VocabType not supported")
      sample["targets"].append(EOS)

      yield sample

  def feature_encoders(self, data_dir):
    """data generation for training"""
    encoders = super().feature_encoders(data_dir)
    feat_encoders = self.get_src_feature_encoders(data_dir)
    for f_id, encoder in enumerate(feat_encoders):
      encoders["sfeats.%d" %f_id] = encoder
    return encoders

  def example_reading_spec(self):
    data_fields, data_items_to_decoders = super().example_reading_spec()
    # add source features
    sfeats = [feat for feat in self.get_feature_encoders() if feat.startswith("sfeats")]
    for sfeat in sfeats:
      data_fields[sfeat] = tf.VarLenFeature(tf.int64)
    return (data_fields, data_items_to_decoders)

  def get_src_feature_encoders(self, data_dir, tmp_dir=None):
    r"""build source feature encoders"""
    feat_encoders = []
    i = 0
    current_path = data_dir+'/'+self.vocab_sfeat_filenames(i)

    if not os.path.isfile(current_path):
      self.get_or_create_src_feature_vocabs(data_dir, tmp_dir)

    # Search for feature vocab files on disc
    while os.path.isfile(current_path):
      feat_encoders.append(text_encoder.TokenTextEncoder(current_path))
      i += 1
      current_path = data_dir+'/'+self.vocab_sfeat_filenames(i)
    return feat_encoders

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.stop_at_eos = int(True)        
    source_vocab_size = self._encoders["inputs"].vocab_size
    p.input_modality = {
      "inputs": (registry.Modalities.SYMBOL, source_vocab_size)
      }
        
    # include source features
    sfeat_nb = len([feat for feat in self.get_feature_encoders() if feat.startswith("sfeats")])
    
    for f_number in range(sfeat_nb):
      sfeat = "sfeats." + str(f_number)
      p.input_modality[sfeat] = ("symbol:sfeature",
                                 {"f_number": f_number,
                                  "vocab_size" : self.get_feature_encoders()[sfeat].vocab_size})
            
      target_vocab_size = self._encoders["targets"].vocab_size
      p.target_modality = (registry.Modalities.SYMBOL, target_vocab_size)
      
      p.sfeat_delimiter = self.sfeat_delimiter
      p.use_subword_tags = self.use_subword_tags
      p.vocab_type = self.vocab_type

      if self.packed_length:
        identity = (registry.Modalities.GENERIC, None)
        if self.has_inputs:
          p.input_modality["inputs_segmentation"] = identity
          p.input_modality["inputs_position"] = identity
        p.input_modality["targets_segmentation"] = identity
        p.input_modality["targets_position"] = identity
    

@registry.register_symbol_modality("sfeature")
class SFeatureSymbolModality(modalities.SymbolModality):
  def __init__(self, model_hparams, feat_hp=None):
    self._model_hparams = model_hparams
    self._vocab_size = feat_hp['vocab_size']    
    self.f_number = feat_hp['f_number']
    try:
      self.sfeat_size = int(model_hparams.get('source_feature_sizes').split(':')[self.f_number])
    except IndexError:
      raise IndexError("Source feature size not provided as hyper-parameter")
  
  @property
  def name(self):
    return "symbol_modality_sfeature_%d_%d_%d" % (self._vocab_size, self.sfeat_size, self.f_number)  
      
  def bottom(self, x):        
    while len(x.get_shape()) < 4:
          x = tf.expand_dims(x, axis=-1)
    return self.bottom_sfeats(x)
    
  def bottom_sfeats(self, x, reuse=None):
    name = "src_feat.%s" % self.f_number
    with tf.variable_scope(name, reuse):
      # Squeeze out the channels dimension.
      x = tf.squeeze(x, axis=3)
      var = self._get_weights(hidden_dim=self.sfeat_size)
      x = common_layers.dropout_no_scaling(
        x, 1.0 - self._model_hparams.symbol_dropout)
      ret = common_layers.gather(var, x)
      if self._model_hparams.multiply_embedding_mode == "sqrt_depth":
        ret *= self._body_input_depth**0.5
      ret *= tf.expand_dims(tf.to_float(tf.not_equal(x, 0)), -1)
      return ret    

    
@registry.register_model
class TransformerSrcFeatures(transformer.Transformer):
  """Transformer model with source features"""
    
  def sfeat_size(self, f_id):
    """source feature vector dimension"""
    return int(self.hparams.source_feature_sizes.split(":")[f_id])

  @property
  def sfeat_number(self):
    """number of source features"""
    return len(self.hparams.source_feature_sizes.split(":"))

  def bottom(self, features):
    """Transform features to feed into body."""
    
    transformed_features = super(TransformerSrcFeatures, self).bottom(features)
    _sfeat_number = len([f for f in transformed_features if f.startswith("sfeats") and not f.endswith("_raw")])
    assert _sfeat_number == self.sfeat_number, \
        "More source features set in hparams than observed in input: %s" %self.hparams.source_feature_sizes

    inputs_with_sfeats = transformed_features["inputs"]
    for f_id in range(self.sfeat_number):
      inputs_with_sfeats = tf.concat([inputs_with_sfeats, transformed_features["sfeats."+str(f_id)]], 3)
      
    assert inputs_with_sfeats.shape[-1] == self.hparams.enc_hidden_size

    transformed_features["inputs"] = inputs_with_sfeats

    return transformed_features

  def shard_sfeatures(self, inputs, features, data_parallelism):
    feat_shards = []
    for inp_shard in inputs:
      feat_shards.append({"inputs": inp_shard})
    source_features = {k:v for k, v in features.items() if k.startswith("sfeats")}
    for feat_name, feature in source_features.items():
      feature = tf.expand_dims(feature, axis=1)          
      if len(feature.shape) < 5:
        feature = tf.expand_dims(feature, axis=4)
      s = common_layers.shape_list(inputs)
      feature = tf.reshape(feature, [s[0] * s[1], s[2], s[3], s[4]])
      for ids, feat_shard in enumerate(self._shard_features({feat_name: feature})[feat_name]):
        feat_shards[ids][feat_name] = feat_shard
    return data_parallelism(self.concatenate_sfeats, feat_shards)
  
  def concatenate_sfeats(self, features):
    """concatenate source embedding and features for decoding"""
    input_modality = self._problem_hparams.input_modality["inputs"]
    with tf.variable_scope(input_modality.name):
      inputs = input_modality.bottom(features["inputs"])

    sfeats = []
    for idx in range(self.sfeat_number):
      sfeat = features["sfeats."+str(idx)]
      input_modality = self._problem_hparams.input_modality["sfeats."+str(idx)]
      with tf.variable_scope(input_modality.name):
        sfeat = input_modality.bottom_sfeats(sfeat)
      inputs = tf.concat([inputs, sfeat], -1)

    return inputs
