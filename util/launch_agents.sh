#!/bin/bash

# --- Configuration ---
GPU_INDICES=(0) 
AGENT_STARTER_SCRIPT="code/DNN/FedQS_hypertune.py" 
SLEEP_TIME=5 

NUM_GPUS=${#GPU_INDICES[@]}

echo "Starting W&B Sweep Agents on the following GPUs: ${GPU_INDICES[@]}"
echo "Total agents to launch: ${NUM_GPUS}"
echo "--------------------------------------------------------"

# --- Loop to Set and Launch ---
for i in "${GPU_INDICES[@]}"; do    
    # This is where the value is SET for the subsequent process
    export CUDA_VISIBLE_DEVICES=$i
    
    echo "Launching process restricted to GPU: $i"
    
    # The python process inherits CUDA_VISIBLE_DEVICES=$i
    python3 ${AGENT_STARTER_SCRIPT} &

    sleep ${SLEEP_TIME}
done
