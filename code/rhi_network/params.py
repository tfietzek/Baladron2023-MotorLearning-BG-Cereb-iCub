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

    # time constants
    'cov_tau': 2200.0,
    'cov_alpha': 2600.0,

    # M1 decision
    'temperature_softmax': 0.025,

    # SNc
    'baseline_dopa': 0.2,
    'regularization_rpe': 0.2,
    'k_burst': 2.2,  # modulatory DA burst when SNc firing

    # connectivity strengths
    'init_w_striatum': 0.0,
    'w_snr': 1.0,
    'w_thalamus': 1.0,
    'w_m1': 1.0,
    'w_cpg': 1.0,
    'w_fb_m1': 1.05,

    # laterals
    'w_strD1_strD1': 0.2,
    'w_snr_snr': 0.35,
    'w_thalamus_thalamus': 0.35,
    'w_m1_m1': 0.25,

    # s1 to striatum parameters
    'reg_threshold_s1': 0.125,
    'reg_threshold_d1': 0.23,
    'alpha_regularization': 0.2,

    # brainstem parameters
    'encodings_m1': np.arange(16, 81, 2, dtype=np.float16),

    # inverse kinematics parameters
    'cpg_weights_file': 'results/RHI_j11_sigma2/network_inverse_kinematic/best_inverse_results.npz',
}
