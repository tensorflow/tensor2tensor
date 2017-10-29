#!/bin/bash

mosesdecoder=$1

targets_file=$2
decodes_file=$3

# Tokenize.
perl $mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < $targets_file > $targets_file.tok
perl $mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < $decodes_file > $decodes_file.tok

# Get rouge scores
python get_rouge.py --decodes_filename $decodes_file.tok --targets_filename $targets_file.tok
