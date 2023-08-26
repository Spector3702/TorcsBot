#!/bin/bash

# Start the Xvfb service
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &

# Wait for a short duration to ensure Xvfb is up
sleep 2

# Run the training script
python TORCS_NEAT/train.py --generations 1

