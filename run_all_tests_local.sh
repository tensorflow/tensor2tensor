# TODO: unify with run_all_tests.sh

#!/usr/bin/env bash

# A rather terrible/fragile workaround
# to gain access to 'dki'
# Probably OK for now, since this all is used strictly to run tests 
# locally...
shopt -s expand_aliases
source ~/diseaseTools/scripts/vm_setup/dev_config/.bashrc_aliases_fathom

dki gcr.io/fathom-containers/t2t_test python3 -m pytest -vv \
       --ignore=/usr/src/t2t/tensor2tensor/utils/registry_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/utils/trainer_lib_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/visualization/visualization_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/problems_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/bin/t2t_trainer_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/data_generators/algorithmic_math_test.py \
       --ignore=/usr/src/t2t/tensor2tensor/models/research/r_transformer_test.py \
       --junitxml=/usr/src/t2t/test_results/pytest/unittests.xml \
       /usr/src/t2t/tensor2tensor
