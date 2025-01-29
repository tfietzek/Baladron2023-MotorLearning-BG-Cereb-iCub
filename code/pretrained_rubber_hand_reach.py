import os
import pandas as pd
from typing import List, Optional

from rhi_network.reaching_model import *
from monitoring import PopMonitor
from train_rubber_hand_reach import simulate_reaching, simulate_cpg
from mlp_utils import merge_training_data, merge_test_data
from run_rubber_hand_reach import normalize_list_column


if __name__ == '__main__':
    theta_vis: float = 50.0
    theta_proprio: float = 30.0
    n_samples: Optional[int] = 2
    make_animations: bool = False

    pretrained_model_path: str = 'results/RHI_j11_sigma2/bg_reach_training/bg_synapses.npz'
    rhi_path: str = 'data_out/data_RHI_jitter_1_1_sigma_prop_2.npz'

    # set up monitors
    monitors = PopMonitor(populations=[S1, StrD1, SNr, VL, M1, SNc, ],
                          sampling_rate=1.0)
    monitors_plot_types: List[str] = ['Bar', 'Bar', 'Bar', 'Bar', 'Bar',
                                      'Line', ]  # extend this when more pops are added
    m1_monitor = ann.Monitor(M1, 'r', )  # to record M1 firing rates for decision

    # compile model
    compile_folder = f'annarchy/pretrained_rubber_hand_reach/'
    if not os.path.exists(compile_folder):
        os.makedirs(compile_folder)
    ann.compile(directory=compile_folder)

    # load pretrained model
    ann.load(pretrained_model_path)

    # training data
    if theta_vis == theta_proprio:
        # load congruent data
        df = merge_training_data(rhi_path=rhi_path,
                                 cpg_path='results/RHI_j11_sigma2/network_inverse_kinematic/best_inverse_results.npz',
                                 save_name='/RHI_j11_sigma2_training.parquet')
        # normalise s1 inputs
        df = normalize_list_column(df, column_name='r_output')

        # get inputs to specific angle
        s1_inputs = df[df['theta'] == theta_proprio]['r_output'].tolist()
    else:
        df = merge_test_data(rhi_path=rhi_path,
                             cpg_path='results/RHI_j11_sigma2/network_inverse_kinematic/best_inverse_results.npz',
                             save_name='/RHI_j11_sigma2_test.parquet')

        # normalise s1 inputs
        df = normalize_list_column(df, column_name='r_output')

        # get inputs to specific angle
        s1_inputs = df[(df['theta'] == theta_proprio) & (df['vision_theta'] == theta_vis)]['r_output'].tolist()

    for i, s1_input in enumerate(s1_inputs):
        save_folder: str = f'results/pretrained_rubber_hand_reach/trial_{i}/'
        os.makedirs(save_folder, exist_ok=True)

        monitors.start()

        m1_rates = simulate_reaching(s1_inputs=s1_input,
                                     m1_inputs=None,
                                     training=False,
                                     wait_time=50.,
                                     reach_time=200.,
                                     m1_monitor=m1_monitor)

        cpg_output, m1_theta = simulate_cpg(m1_rates=m1_rates,
                                            temperature=parameters['temperature_softmax'],
                                            encodings_m1=parameters['encodings_m1'],
                                            regularization=0.0,)

        monitors.stop()

        if make_animations:
            # save animation
            monitors.animate_current_monitors(save_name=save_folder + 'animation.mp4',
                                              plot_types=monitors_plot_types,
                                              fig_size=(20, 20),
                                              t_init=0,
                                              clear_monitors=False)

        # save monitors
        monitors.save(save_folder, delete=True)

        # save other data
        with open(save_folder + 'cpg_output.txt', 'w') as f:
            f.write(str(cpg_output))

        with open(save_folder + 'm1_theta.txt', 'w') as f:
            f.write(str(m1_theta))

        if n_samples is not None:
            if i >= n_samples:
                break

    # clear monitors
    m1_monitor.stop()
