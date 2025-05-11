#!/bin/bash
# This script runs the specified simulation in Python.
# It checks for a virtual environment, activates it, and runs the appropriate simulation based on user input.

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if an argument was provided
if [ $# -eq 0 ]; then
    echo "Usage: ./run.sh [bus|projectile|ex1|ex2] [additional arguments]"
    exit 1
fi

# Extract the first argument (simulation type)
SIM_TYPE=$1
shift  # Remove the first argument, leaving any additional args

# Run the appropriate simulation
case $SIM_TYPE in
    bus|ex1)
        echo "Running bus maintenance simulation..."
        echo "Note: Both 'bus' and 'ex1' run the bus maintenance simulation"
        echo -e "\n"
        python src/bus/main.py "$@"
        ;;
    projectile|ex2)
        echo "Running projectile motion simulation..."
        echo "Note: Both 'projectile' and 'ex2' run the projectile motion simulation"
        echo -e "\n"
        python src/projectile/main.py "$@"
        ;;
    *)
        echo "Unknown simulation type: $SIM_TYPE"
        echo "Usage: ./run.sh [bus|projectile|ex1|ex2] [additional arguments]"
        exit 1
        ;;
esac