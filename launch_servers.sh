#!/bin/bash

# Total number of GPUs
TOTAL_GPUS=8

# Number of servers per GPU
SERVERS_PER_GPU=9

# Starting port number
BASE_PORT=8000

# Loop over each GPU
for ((gpu=0; gpu<$TOTAL_GPUS; gpu++)); do
  # Loop over each server for the current GPU
  for ((server=0; server<$SERVERS_PER_GPU; server++)); do
    # Calculate the port number for each server instance
    port=$((BASE_PORT + gpu * SERVERS_PER_GPU + server))
    
    # Launch the server on the specified GPU and port in the background
    CUDA_VISIBLE_DEVICES=$gpu uvicorn instructmultilingual.server:app --host 0.0.0.0 --port $port &
    
    echo "Launched server on GPU $gpu at port $port"
  done
done

# Wait for all background jobs to finish (optional)
wait

