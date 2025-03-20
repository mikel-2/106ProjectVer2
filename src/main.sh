#!/bin/bash

echo "Running EZ Diffusion Model Simulation..."

# Run the Python script
python3 src/ez_diffusion.py > data/results.csv

echo "Results saved to data/results.csv"
