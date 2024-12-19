#!/bin/bash

# Function to convert seconds to days:hours:minutes format
format_time() {
    local seconds=$1
    local days=$((seconds/86400))
    local hours=$(( (seconds%86400)/3600 ))
    local minutes=$(( (seconds%3600)/60 ))
    echo "${days}:${hours}:${minutes}"
}

# Record start time
start_time=$(date +%s)

# Run simulations in parallel
python run_rubber_hand_reach.py --data_set RHI_j11_sigma2 --rhi_data_path data_out/data_RHI_jitter_1_1_sigma_prop_2.npz --temperature 0.5 &
python run_rubber_hand_reach.py --data_set RHI_j12_sigma4 --rhi_data_path data_out/data_RHI_jitter_1_2_sigma_prop_4.npz --temperature 0.5 &

# Wait for all processes to finish
wait

# Record end time
parallel_end_time=$(date +%s)
parallel_duration=$((parallel_end_time - start_time))
parallel_runtime=$(format_time $parallel_duration)

echo "Parallel run completed in $parallel_runtime"