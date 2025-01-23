#!/bin/bash

# Run optimization for RHI_j11_sigma2
python optuna_rhi_model_tuning.py \
    --data_set="RHI_j11_sigma2" \
    --storage="sqlite:///optuna_results_RHI_j11_sigma2.db" \
    --n_trials=120 &

# Run optimization for RHI_j12_sigma4
python optuna_rhi_model_tuning.py \
    --data_set="RHI_j12_sigma4" \
    --storage="sqlite:///optuna_results_RHI_j12_sigma4.db" \
    --n_trials=120 &

# Wait for both processes to complete
wait