# SO-SIMULATION-MP

## Description

This project is a simulation miniproject developed for the Simulation and Optimization course. It implements various simulation techniques and optimization algorithms to solve practical problems.

## Table of Contents

- [Description](#description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Authors](#authors)

## Project Structure

```
SO-SIMULATION-MP/
├── assets/
│   ├── report.pdf                   # Project report document
│   ├── report_source_files.zip      # Source files used to generate the report
│   └── simopt2425_SimProject.pdf    # Project specification document
├── config/
│   ├── bus_config.yaml              # Configuration template for bus maintenance simulation
│   └── projectile_config.yaml       # Configuration template for projectile motion simulation
├── results/
│   ├── bus/                         # Output directory for bus simulation results
│   │   └── [simulation outputs]
│   └── projectile/                  # Output directory for projectile simulation results
│       └── [simulation outputs]
├── src/
│   ├── bus/                         # Bus maintenance simulation module
│   │   ├── __init__.py              # Package initialization
│   │   ├── config.py                # Configuration loader and validator
│   │   ├── experiments.py           # Experiment definitions and runners
│   │   ├── main.py                  # Entry point for bus simulation
│   │   ├── simulation.py            # Core simulation logic
│   │   └── visualization.py         # Result plotting and visualization
│   └── projectile/                  # Projectile motion simulation module
│       ├── __init__.py              # Package initialization
│       ├── analysis.py              # Statistical analysis of simulation results
│       ├── config.py                # Configuration loader and validator
│       ├── main.py                  # Entry point for projectile simulation
│       ├── simulation.py            # Core simulation logic
│       └── visualization.py         # Result plotting and visualization
├── .gitignore                       # Git ignore file
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
├── run.sh                           # Execution script
└── setup.sh                         # Environment setup script
```

## Installation

### Option 1: Using the Setup Script

The easiest way to set up the environment is to use the provided setup script:

```bash
# Clone the repository
git clone https://github.com/Sytuz/SO-Simulation-MP.git
cd SO-Simulation-MP

# Make the setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

The setup script will:
- Create a virtual environment
- Install all required dependencies
- Make the run script executable

### Option 2: Manual Setup

If you prefer to set up manually:

```bash
# Clone the repository
git clone https://github.com/Sytuz/SO-Simulation-MP.git
cd SO-Simulation-MP

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Linux/macOS
source venv/bin/activate
# On Windows
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Option 1: Using the Run Script

The project includes a convenient run script that automatically activates the virtual environment and executes the selected simulation:

```bash
# Make the run script executable (should be done by setup.sh)
#chmod +x run.sh

# Run the bus maintenance simulation
./run.sh bus [additional arguments]
# or
./run.sh ex1 [additional arguments]

# Run the projectile motion simulation
./run.sh projectile [additional arguments]
# or
./run.sh ex2 [additional arguments]
```

### Option 2: Running Python Scripts Directly

If you prefer to run the Python scripts directly:

```bash
# Activate the virtual environment first
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate   # On Windows

# Run the bus maintenance simulation
python src/bus/main.py [additional arguments]

# Run the projectile motion simulation
python src/projectile/main.py [additional arguments]
```

## Authors

<table>
  <tr>
    <td align="center">
        <a href="https://github.com/Sytuz">
            <img src="https://avatars0.githubusercontent.com/Sytuz?v=3" width="100px;" alt="Alexandre"/>
            <br />
            <sub>
                <b>Alexandre Ribeiro</b>
                <br>
                <i>108122</i>
            </sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/miguel-silva48">
            <img src="https://avatars0.githubusercontent.com/miguel-silva48?v=3" width="100px;" alt="Miguel"/>
            <br />
            <sub>
                <b>Miguel Pinto</b>
                <br>
                <i>107449</i>
            </sub>
        </a>
    </td>
  </tr>
</table>