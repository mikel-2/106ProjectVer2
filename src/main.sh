#!/bin/bash

echo "🖖 Running EZ Diffusion Model Simulation..."

PYTHONPATH=src python3 src/simulation.py

echo "✅ Results saved to data/results.csv"
