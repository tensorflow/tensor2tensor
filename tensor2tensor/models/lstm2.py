"""Minimal recreation of problem.
"""
from tensor2tensor.utils import registry
from tensor2tensor.layers import common_hparams

@registry.register_hparams
def lstm2():
    """Hparams for minimal example, copied from T2T LSTM hparams."""
    hparams = common_hparams.basic_params1()
    hparams.batch_size = 1024
    hparams.hidden_size = 128
    hparams.num_hidden_layers = 2
    # uncomment this line to fix things
    # hparams.initializer = "uniform_unit_scaling"
    return hparams
    
# To reproduce problem, run like this:

# python3 t2t-trainer --data_dir <DATA_DIR> --hparams_set lstm2
# --local_eval_frequency 1 --model lstm_seq2seq --output_dir
# <OUTPUT_DIR> --problem translate_ende_wmt8k --tmp_dir <TMP_DIR>
