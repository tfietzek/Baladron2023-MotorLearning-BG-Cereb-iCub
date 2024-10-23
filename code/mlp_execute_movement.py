import importlib
import os.path
from typing import Optional
import pandas as pd
import numpy as np
import gc

# Model
from kinematic import *
from cpg import *
from train_BG_reaching import execute_movement, random_goal2_iCub
from mlp_inverse_fit import load_training_rhi_thetas, save_mlp

# CPG
import CPG_lib.parameter as params
from CPG_lib.MLMPCPG.MLMPCPG import *
from CPG_lib.MLMPCPG.myPloting import *
from CPG_lib.MLMPCPG.SetTiming import *
from scipy.optimize import minimize

from mlp_inverse_fit import train_mlp, test_mlp, train_mlps, merge_training_data, merge_test_data, get_prediction, make_reaching_error_data



if __name__ == '__main__':
    data_set = (0, 1)[int(sys.argv[1])]
    hidden_layer_sizes = ((64, 64,), (128, 128,), (128, 64,), (256, 256,), (256, 128,), (256, 64,),)[int(sys.argv[2])]

    # data paths
    inv_path = ('results/RHI_j11_sigma2/network_inverse_kinematic/best_inverse_results_run.npz',
                'results/RHI_j12_sigma4/network_inverse_kinematic/best_inverse_results_run.npz')[data_set]
    rhi_path = ('data_out/data_RHI_jitter_1_1_sigma_prop_2.npz',
                'data_out/data_RHI_jitter_1_2_sigma_prop_4.npz')[data_set]
    data_name = ('RHI_j11_sigma2', 'RHI_j12_sigma4')[data_set]

    if not os.path.isfile(inv_path):
        make_reaching_error_data(cpg_path=inv_path)

    df_train = merge_training_data(rhi_path=rhi_path, cpg_path=inv_path)
    df_test = merge_test_data(rhi_path=rhi_path, cpg_path=inv_path)

    for hidden_layer_size in hidden_layer_sizes:
        # save results in this folder
        folder = f'results/{data_name}/mlp_shape[{hidden_layer_size[0], hidden_layer_size[1]}]/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        # train mlp
        print(f'Training MLP with hidden layer size: {hidden_layer_size}')
        mlp, scaler, onehot = train_mlps(df_train,
                                 hidden_layer_size=hidden_layer_size,
                                 max_iter=5,
                                 test_size=0.2)

        save_mlp(mlp, scaler, save_path=folder)

