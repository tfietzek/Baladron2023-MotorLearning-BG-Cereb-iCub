import numpy as np
import os


def weights_to_cpg(file: str,
                   cpg_dim: int) -> np.ndarray:
    if not os.path.isfile(file):
        raise Exception(f"File {file} does not exist. Please run run_inverse_kinematics.py first to generate "
                        f"the connection weights.")
    else:
        cpg_data = np.load(file)

    w = cpg_data['cpg_params_to_goals'].reshape(-1, cpg_dim)

    # ANNarchy expects the weights in post to pre order.
    return w.T


if __name__ == '__main__':
    weights_to_cpg(file='../results/RHI_j11_sigma2/network_inverse_kinematic/best_inverse_results.npz', cpg_dim=24)