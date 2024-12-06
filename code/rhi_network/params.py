"""
Parameters for reaching model of the basal ganglia
"""
import numpy as np

parameters = {
    # dimensions
    'dim_s1': 50,
    'dim_bg': 33,
    'dim_cpg': 24,

    # baseline values
    'baseline_snr': 1.0,
    'baseline_thalamus': 1.0,
    'baseline_dopa': 0.2,

    # connectivity strengths
    'init_w_striatum': 0.0,
    'w_snr': 1.0,
    'w_thalamus': 1.0,
    'w_m1': 1.0,
    'w_cpg': 1.0,

    # brainstem parameters
    'softmax_temperature': 1.0,
    'encodings_m1': np.arange(15, 80, 2, dtype=np.float16),

    # inverse kinematics parameters
    'cpg_weights_file': 'results/RHI_j11_sigma2/network_inverse_kinematic/best_inverse_results.npz'
}

