import os
import pandas as pd

from rhi_network.reaching_model import *
from monitoring import PopMonitor
from train_rubber_hand_reach import simulate_reaching, predict_over_inputs
from mlp_utils import merge_training_data


if __name__ == '__main__':
    n_samples: int = 50  # subset of data to visualise
    pretrained_model_path: str = 'results/RHI_j11_sigma2/bg_reach_training/bg_synapses.npz'

    # set up monitors
    monitors = PopMonitor(populations=[S1, M1, StrD1, SNc, VL, SNr],
                          sampling_rate=1.0)

    # compile model
    compile_folder = f'annarchy/pretrained_rubber_hand_reach/'
    if not os.path.exists(compile_folder):
        os.makedirs(compile_folder)
    ann.compile(directory=compile_folder)

    # load pretrained model
    ann.load(pretrained_model_path)

    # training data
    df_train = merge_training_data(rhi_path='data_out/data_RHI_jitter_1_1_sigma_prop_2.npz',
                                   cpg_path='results/RHI_j11_sigma2/network_inverse_kinematic/best_inverse_results.npz',
                                   save_name='/RHI_j11_sigma2_training.parquet').sample(n_samples)

    s1_inputs = df_train['r_output'].tolist()

    # simulate
    monitors.start()

    for s1_input in s1_inputs:
        simulate_reaching(s1_inputs=s1_input,
                          m1_inputs=None,
                          training=False,
                          wait_time=50.,
                          reach_time=150.)

    monitors.stop()
    monitors.animate_current_monitors()
