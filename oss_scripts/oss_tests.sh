#!/bin/bash

set -v  # print commands as they're executed

# Instead of exiting on any failure with "set -e", we'll call set_status after
# each command and exit $STATUS at the end.
STATUS=0
function set_status() {
    local last_status=$?
    if [[ $last_status -ne 0 ]]
    then
      echo "<<<<<<FAILED>>>>>> Exit code: $last_status"
    fi
    STATUS=$(($last_status || $STATUS))
}

# Check env vars set
echo "${TF_VERSION:?}" && \
echo "${TF_LATEST:?}" && \
echo "${TRAVIS_PYTHON_VERSION:?}"
set_status
if [[ $STATUS -ne 0 ]]
then
  exit $STATUS
fi

# Check import
python -c "from tensor2tensor.models import transformer; print(transformer.Transformer.__name__)"
set_status

# Run tests
# Ignores:
# Tested separately:
#   * registry_test
#   * trainer_lib_test
#   * visualization_test
#   * trainer_model_based_test
#   * allen_brain_test
#   * models/research
# algorithmic_math_test: flaky
# test_utils.py is not a test, but pytest thinks it is.
# subword_text_encoder_ops_test, pack_sequences_ops_test: interface with C++ ops
# trax tests need C++
# TODO(davidso): Re-enable EvolvedTransformer when possible.
# others (see below) enable eager, so can't be tested along with the others in
# pytest
pytest --disable-warnings \
  --ignore=tensor2tensor/bin/t2t_trainer_test.py \
  --ignore=tensor2tensor/data_generators/algorithmic_math_test.py \
  --ignore=tensor2tensor/data_generators/allen_brain_test.py \
  --ignore=tensor2tensor/data_generators/ops/pack_sequences_ops_test.py \
  --ignore=tensor2tensor/data_generators/ops/subword_text_encoder_ops_test.py \
  --ignore=tensor2tensor/data_generators/problem_test.py \
  --ignore=tensor2tensor/envs/mujoco_problems_test.py \
  --ignore=tensor2tensor/envs/rendered_env_problem_test.py \
  --ignore=tensor2tensor/layers/bayes_test.py \
  --ignore=tensor2tensor/layers/common_attention_test.py \
  --ignore=tensor2tensor/layers/common_layers_test.py \
  --ignore=tensor2tensor/layers/common_video_test.py \
  --ignore=tensor2tensor/layers/discretization_test.py \
  --ignore=tensor2tensor/layers/latent_layers_test.py \
  --ignore=tensor2tensor/layers/modalities_test.py \
  --ignore=tensor2tensor/layers/ngram_test.py \
  --ignore=tensor2tensor/layers/reversible_layers_test.py \
  --ignore=tensor2tensor/models/evolved_transformer_test.py \
  --ignore=tensor2tensor/models/research \
  --ignore=tensor2tensor/models/video/nfg_conv3d_test.py \
  --ignore=tensor2tensor/models/video/nfg_conv_lstm_test.py \
  --ignore=tensor2tensor/models/video/nfg_conv_test.py \
  --ignore=tensor2tensor/models/video/nfg_uncond_test.py \
  --ignore=tensor2tensor/rl \
  --ignore=tensor2tensor/trax \
  --ignore=tensor2tensor/utils/registry_test.py \
  --ignore=tensor2tensor/utils/t2t_model_test.py \
  --ignore=tensor2tensor/utils/test_utils.py \
  --ignore=tensor2tensor/utils/test_utils_test.py \
  --ignore=tensor2tensor/utils/trainer_lib_test.py \
  --ignore=tensor2tensor/visualization/visualization_test.py \
  --deselect=tensor2tensor/utils/beam_search_test.py::BeamSearchTest::testTPUBeam
set_status

# TODO(afrozm): Enable trax tests they currently need GLIBCXX_3.4.21
# Travis Error:
# ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.21' not found (required by /home/travis/virtualenv/python3.6.3/lib/python3.6/site-packages/jaxlib/_pywrap_xla.so)

# These tests enable eager, so are tested separately.
pytest --disable-warnings \
  tensor2tensor/data_generators/problem_test.py \
  tensor2tensor/layers/bayes_test.py \
  tensor2tensor/layers/common_attention_test.py \
  tensor2tensor/layers/common_layers_test.py \
  tensor2tensor/layers/common_video_test.py \
  tensor2tensor/layers/discretization_test.py \
  tensor2tensor/layers/latent_layers_test.py \
  tensor2tensor/layers/modalities_test.py \
  tensor2tensor/layers/ngram_test.py \
  tensor2tensor/layers/reversible_layers_test.py \
  tensor2tensor/utils/t2t_model_test.py \
  tensor2tensor/utils/test_utils_test.py \
  --deselect=tensor2tensor/layers/common_video_test.py::CommonVideoTest::testGifSummary
set_status

pytest --disable-warnings tensor2tensor/utils/registry_test.py
set_status

pytest --disable-warnings tensor2tensor/utils/trainer_lib_test.py
set_status

pytest --disable-warnings tensor2tensor/visualization/visualization_test.py
set_status

pytest --disable-warnings tensor2tensor/data_generators/allen_brain_test.py
set_status


# Test models/research only against tf-nightly
if [[ "$TRAVIS_PYTHON_VERSION" == "2.7"  ]] && [[ "$TF_VERSION" == "tf-nightly"  ]]
then
  # Ignores:
  # * Glow requires the CIFAR-10 dataset to be generated
  pytest --disable-warnings tensor2tensor/models/research \
    --ignore=tensor2tensor/models/research/glow_test.py
  set_status
fi

if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]] && [[ "$TF_VERSION" == "$TF_LATEST"  ]]
then
    # TODO(afrozm): Once we drop support for 1.10 we can get rid of this.
    pytest --disable-warnings \
      tensor2tensor/utils/beam_search_test.py::BeamSearchTest::testTPUBeam
    set_status
    # TODO(afrozm): Enable other tests in the RL directory.
    # Can't add disable warning here since it parses flags.
    pytest tensor2tensor/rl/trainer_model_based_test.py
    set_status
    jupyter nbconvert --ExecutePreprocessor.timeout=600 --to notebook --execute tensor2tensor/notebooks/hello_t2t.ipynb
    set_status
    jupyter nbconvert --ExecutePreprocessor.timeout=600 --to notebook --execute tensor2tensor/notebooks/t2t_problem.ipynb;
    set_status
fi

# Test --t2t_usr_dir
t2t-trainer --registry_help --t2t_usr_dir=./tensor2tensor/test_data/example_usr_dir 2>&1 | grep my_very_own_hparams && echo passed
set_status

exit $STATUS
