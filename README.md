# RIE Singular Value Cleaning of Lagged Covariance Matrices

## Overview
This repository contains the code for Bryan's COMP0029 final year project, which focuses on the singular value cleaning of lagged covariance matrices. The project implements and visualises the effectiveness of the Rotationally Invariant Estimator (RIE) method in statistical analysis of high-dimensional time series data.

## Getting Started

### Prerequisites
Ensure you have Python installed on your machine (Python 3.6 or later is recommended). You will also need the following libraries:
- NumPy
- Matplotlib
- SciPy
- Pandas
- Scikit-Learn
- Statsmodels

You can install these packages using pip by creating a `requirements.txt` file with all the needed packages and then running:
```bash
pip install -r requirements.txt
```

## Configuration and Usage

### RIE Visualisations
Adjust the configuration settings in `RIE_visualisations.py` according to your data and preferences. The settings include:
- Figure size
- Whether to save or display the figures
- The models to be analysed

To generate visualisations that demonstrate the cleaning process and effectiveness of the RIE method, run the script from the command line:

```bash
python RIE_visualisations.py
```

### Phi Distribution Calculation
To test if the singular values convert to the correct phi values run the phi_AR.py script. This script checks if the re-calculated phi values are equal to the original phi values:

```bash
python phi_AR.py
```