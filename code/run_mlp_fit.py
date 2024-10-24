import os
import sys

from mlp_utils import (train_mlp,
                       test_mlp,
                       train_mlps,
                       merge_training_data,
                       merge_test_data,
                       get_prediction,
                       save_mlp)



if __name__ == '__main__':
    data_set = (0, 1)[int(sys.argv[1])]
    hidden_layer_size = ((64,), (128,), (256,), (512,))[int(sys.argv[2])]

    # data paths
    inv_path = ('results/RHI_j11_sigma2/network_inverse_kinematic/best_inverse_results.npz',
                'results/RHI_j12_sigma4/network_inverse_kinematic/best_inverse_results.npz')[data_set]
    rhi_path = ('data_out/data_RHI_jitter_1_1_sigma_prop_2.npz',
                'data_out/data_RHI_jitter_1_2_sigma_prop_4.npz')[data_set]
    data_name = ('RHI_j11_sigma2', 'RHI_j12_sigma4')[data_set]

    df_train = merge_training_data(rhi_path=rhi_path, cpg_path=inv_path)
    df_test = merge_test_data(rhi_path=rhi_path, cpg_path=inv_path)

    # save results in this folder
    folder = f'results/{data_name}/mlp_shape[{hidden_layer_size[0]}]/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    # train mlp
    print(f'Training MLP with hidden layer size: {hidden_layer_size}')
    mlp, scaler, onehot = train_mlp(df_train, hidden_layer_size=hidden_layer_size, test_size=0.2)

    save_mlp(mlp, scaler, save_path=folder)

