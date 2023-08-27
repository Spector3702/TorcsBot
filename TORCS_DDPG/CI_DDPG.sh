#!/bin/bash

# Start the Xvfb service
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &

# Wait for a short duration to ensure Xvfb is up
sleep 2

# Run the training script
python TORCS_DDPG/train.py --device cpu --episodes 1

# Run the testing script
python TORCS_DDPG/test.py --device cpu
