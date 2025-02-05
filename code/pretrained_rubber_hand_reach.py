import os
import pandas as pd
from typing import List, Optional

from rhi_network.reaching_model import *
from monitoring import PopMonitor
from train_rubber_hand_reach import simulate_reaching, simulate_cpg
from mlp_utils import merge_training_data, merge_test_data
from run_rubber_hand_reach import normalize_list_column


if __name__ == '__main__':
    # parameters
    ids = {
        'training': (1, 2),
        'test': (1, 4),
    }

    theta_proprio: float = 0.0
    make_animations: bool = False

    pretrained_model_path: str = 'results/RHI_j11_sigma2/bg_reach_training/bg_synapses.npz'

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

    # load congruent data
    df_train = pd.read_parquet('results/RHI_j11_sigma2/bg_reach_training/test_results.parquet')
    # normalise s1 inputs
    df_train = normalize_list_column(df_train, column_name='r_output')

    # load incongruent data
    df_test = pd.read_parquet('results/RHI_j11_sigma2/bg_reach_testing/test_results.parquet')
    # normalise s1 inputs
    df_test = normalize_list_column(df_test, column_name='r_output')

    # get data from a given row index
    df_train = df_train.iloc[list(ids['training'])]
    df_test = df_test.iloc[list(ids['test'])]

    # get s1 inputs
    s1_inputs = df_train['r_output'].tolist()

    # append test data s1 inputs
    s1_inputs.extend(df_test['r_output'].tolist())

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

    # clear monitors
    m1_monitor.stop()
