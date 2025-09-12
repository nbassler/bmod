# Beam Model Generation (bmod)

A Python toolkit for analyzing beam profile measurements from XRV4000 systems and extracting beam optics parameters.

<img width="2565" height="1638" alt="fit_plot_cubic_energy_180 0" src="https://github.com/user-attachments/assets/d8175c9c-af5b-415f-bf98-a9234eed543a" />

## Table of Contents
1. [Installation](#installation)
2. [Workflow Overview](#workflow-overview)
3. [Data Structure Requirements](#data-structure-requirements)
4. [Configuration](#configuration)
5. [Running the Analysis](#running-the-analysis)
6. [Output Interpretation](#output-interpretation)
7. [Mathematical Models](#mathematical-models)

## Installation

```bash
git clone https://github.com/your-repo/bmod.git
cd bmod
pip install -r requirements.txt
```

## Workflow Overview

The analysis consists of two main steps:

1. **Image Processing**: Extract beam sizes from profile images
2. **Twiss Parameter Extraction**: Fit models to beam size evolution

## Data Structure Requirements

### Directory Structure
```
data_root/
├── position_001/  # First measurement position
│   ├── energy_001.tif
│   ├── energy_002.tif
│   └── ...
├── position_002/
│   ├── energy_001.tif
│   └── ...
└── ...
```

### Naming Convention
- Directories should be named with sequential numbers (or sortable names)
- Image files should be named with sequential energy identifiers
- Order of directories and files must match configuration file

## Configuration

Create a `bmod.toml` file in your working directory:

```toml
# Example configuration file

# Positions in mm (order must match directory numbering)
zpos = [-100.0, -50.0, 0.0, 50.0, 100.0, 150.0, 200.0, 250.0]

# Energies in MeV (order must match file numbering)
energies = [
    244, 240, 230, 220, 210, 200, 190, # field 1
    180, 170, 160, 150, 140, 130, 120, 115, 110, # field 2
    105, 100, 95, 90, 85, 80, 75, 70,  # field 3
]

# Reference position (mm) for fitting
z0 = -500.0
```

## Running the Analysis

### Step 1: Image Processing

```bash
PYTHONPATH=. python3 bmod/main.py -v /path/to/images -o output.csv
```

This will:
1. Process all images in the specified directory
2. Fit 2D Gaussians to each beam profile
3. Output a CSV file (`output.csv`) containing beam sizes (σ_x, σ_y) at each position and energy

### Step 2: Twiss Parameter Extraction

```bash
PYTHONPATH=. python3 bmod/xrv_twiss_main.py -v output.csv twiss_results.csv
```

This will:
1. Read the beam size data from the input CSV
2. Perform both quadratic and cubic fits to extract Twiss parameters
3. Generate plots of the fits for each energy
4. Output a combined CSV file with all results

Options:
- `--z0`: Override the reference position (default: -500.0 mm)
- `-v`: Increase verbosity (use `-vv` for debug level)

## Output Interpretation

### Output Files

1. **Intermediate CSV**: Beam sizes at each position and energy
   - Columns: `z`, `energy`, `sigma_x_mm`, `sigma_y_mm`

2. **Final Results CSV**: Contains both quadratic and cubic fit results
   - Prefixes: `quad_` for quadratic fit parameters, `cubic_` for cubic fit parameters
   - Key derived parameters for each plane (x and y):
     - Beam size at reference position (`x`, `y`)
     - Beam divergence (`x'`, `y'`)
     - Beam correlation (`xx'`, `yy'`)

3. **Plot Files**: Visualizations of the fits for each energy level

### Fit Parameters

For both models, the parameters relate to Twiss parameters as follows:

| Parameter | Description | Relation to Twiss |
|-----------|-------------|-------------------|
| a         | Quadratic term | ∝ ε/β (emittance/beta) |
| b         | Linear term | ∝ -2α (alpha parameter) |
| c         | Constant term | ∝ εβ (emittance × beta) |
| d         | Cubic term (cubic only) | Scattering term magnitude |

## Mathematical Models

### Quadratic Model (Vacuum Propagation)

σ² = a·(z-z₀)² + b·(z-z₀) + c

Represents ideal beam propagation in vacuum where:
- σ is the beam size
- z is the longitudinal position
- z₀ is the reference position (default: -500 mm)

### Cubic Model (With Scattering)

σ² = a·(z-z₀)² + b·(z-z₀) + c + d·(z-z₀)³

Extends the quadratic model to account for scattering effects in air, which become more significant at larger distances from the reference point.


## Theoretical Background

The quadratic model is derived from the beam envelope equation in vacuum, where the beam size evolution is governed by the Courant-Snyder parameters:

σ²(z) = εβ(z) = ε(β₀ + (2α₀ + z/β₀)z + γ₀z²)

Where:
- ε is the emittance (a measure of beam quality)
- β₀, α₀, γ₀ are the Courant-Snyder parameters at the reference position

The cubic term accounts for multiple scattering effects in air, which cause the beam size to grow faster than quadratically with distance. This effect is more significant for:
- Lower energy beams
- Longer propagation distances through air


## License

[MIT License](LICENSE)
```
