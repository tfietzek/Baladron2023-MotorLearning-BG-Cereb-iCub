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

# Run python script with argument 0 in background
python run_mlp_fit.py 0 &
pid0=$!

# Run python script with argument 1 in background
python run_mlp_fit.py 1 &
pid1=$!

# Wait for both background processes to complete
wait $pid0
echo "Process with argument 0 finished"
end_time0=$(date +%s)
duration0=$((end_time0 - start_time))
echo "Duration for argument 0: $(format_time $duration0) (days:hours:minutes)"

wait $pid1
echo "Process with argument 1 finished"
end_time1=$(date +%s)
duration1=$((end_time1 - start_time))
echo "Duration for argument 1: $(format_time $duration1) (days:hours:minutes)"

# Calculate total duration
total_duration=$((end_time1 > end_time0 ? duration1 : duration0))
echo "Total duration: $(format_time $total_duration) (days:hours:minutes)"