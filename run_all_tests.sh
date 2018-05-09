docker run -it \
       -v /home/circleci/diseaseTools:/usr/src/diseaseTools \
       -v /home/circlci/tensor2tensor:/usr/src/tensor2tensor \
       -w /usr/src/tensor2tensor \
       gcr.io/fathom-containers/t2tgpu \
       python3 -m pytest \
       --ignore=tensor2tensor/utils/registry_test.py \
       --ignore=tensor2tensor/utils/trainer_lib_test.py \
       --ignore=tensor2tensor/visualization/visualization_test.py \
       --ignore=tensor2tensor/problems_test.py \
       --ignore=tensor2tensor/bin/t2t_trainer_test.py \
       --ignore=tensor2tensor/data_generators/algorithmic_math_test.py \
       --ignore=tensor2tensor/models/research/r_transformer_test.py
