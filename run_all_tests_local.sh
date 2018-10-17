# TODO: unify with run_all_tests.sh

#!/usr/bin/env bash

# A rather terrible/fragile workaround
# to gain access to 'dki'
# Probably OK for now, since this all is used strictly to run tests 
# locally...
shopt -s expand_aliases
source ~/diseaseTools/scripts/vm_setup/dev_config/.bashrc_aliases_fathom

# Base line tests (inc. what to skip) derived from t2t's travis):
# https://github.com/tensorflow/tensor2tensor/blob/master/.travis.yml#L55

# FROM t2t travis (20180803):
#
# Run tests
# Ignores:
# Tested separately:
#   * registry_test
#   * trainer_lib_test
#   * visualization_test
#   * model_rl_experiment_test
#   * allen_brain_test
#   * model_rl_experiment_stochastic_test
#   * models/research
# algorithmic_math_test: flaky
# universal_transformer_test: requires new feature in tf.foldl (rm with TF 1.9)

# Our changes:
# * ignore all of /rl, since we aren't using this (and don't have gym in our Docker image)
# * skip problems_test.py (??why??)
# * skip gym_problems (gym not in our image)
# * skip checkpoint_compatibility_test.py (no tqdm; undo this at some point and just install tqdm in image)
# * skip tensor2tensor/models/research/next_frame_test.py b/c not working but clearly experimental on t2t side
# * skip glow_test which requires cifar dataset
#     https://github.com/tensorflow/tensor2tensor/blob/3f43417310101859f95b74587ffc3686714cc58a/oss_scripts/oss_tests.sh#L71
# * skip common_video_test.py::CommonVideoTest::testGifSummary because of ffmpeg dependency
# * skip tensor2tensor/data_generators/image_utils_test.py because of matplotlib dependency
# * skip tensor2tensor/layers/common_video_test.py because of ffmpeg dependency

dki gcr.io/fathom-containers/t2t_test python3 -m pytest -vv \
       --ignore=/usr/src/app/api-flask/ \
       --ignore=/usr/src/t2t/tensor2tensor/utils/registry_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/utils/trainer_lib_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/visualization/visualization_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/bin/t2t_trainer_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/data_generators/algorithmic_math_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/rl/ \
	   --ignore=/usr/src/t2t/tensor2tensor/data_generators/allen_brain_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/problems_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/data_generators/gym_problems_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/utils/checkpoint_compatibility_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/models/research/next_frame_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/rl/trainer_model_based_stochastic_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/rl/trainer_model_based_sv2p_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/models/research/glow_test.py \
       --deselect=/usr/src/t2t/tensor2tensor/layers/common_video_test.py::CommonVideoTest::testGifSummary \
       --ignore=/usr/src/t2t/tensor2tensor/data_generators/image_utils_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/layers/common_video_test.py \
       --junitxml=/usr/src/t2t/test_results/pytest/unittests.xml \
       /usr/src/t2t/tensor2tensor/

#       /usr/src/t2t/tensor2tensor/models/research/universal_transformer_test.py
#       --ignore=/usr/src/t2t/tensor2tensor/models/research/next_frame_test.py \

dki -w /usr/src/t2t gcr.io/fathom-containers/t2t_test python3 -m pytest -vv \
       /usr/src/t2t/tensor2tensor/utils/registry_test.py

# cdb: I believe we break this because of some minor custom changes; should re-visit
# and verify this at some point.
#dki gcr.io/fathom-containers/t2t_test python3 -m pytest \
#       /usr/src/t2t/tensor2tensor/utils/trainer_lib_test.py

# As-is, requires tqdm.  Commenting out for now; we could consider dropping into docker imag.
#dki gcr.io/fathom-containers/t2t_test python3 -m pytest -vv \
#       /usr/src/t2t/tensor2tensor/visualization/visualization_test.py

# TODO: add allen_brain_test.py once we update t2t
