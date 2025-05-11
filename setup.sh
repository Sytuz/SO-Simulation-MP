#!/bin/bash
# This script sets up the environment for running simulations in Python.
# It creates a virtual environment, installs dependencies, and provides instructions for running simulations.
echo "Setting up the simulation environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete! To run simulations:"
echo "  ./run.sh bus        - Run bus simulation"
echo "  ./run.sh projectile - Run projectile simulation"

# Make run.sh executable
chmod +x run.sh

echo "Environment setup complete!"