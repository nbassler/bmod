# Beam Model Generation (bmod)

A Python toolkit for analyzing beam profile measurements from XRV4000 systems and extracting beam optics parameters.

<img width="2539" height="1638" alt="image" src="https://github.com/user-attachments/assets/7f55df9d-747a-4286-b5d2-a3cccf72ca11" />



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
git clone https://github.com/nbassler/bmod.git
cd bmod
pip install -e .
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

Create a `bmod.toml` file in your working directory.
An example configuration (including air-scan settings) is available at
[bmod_air.toml](https://github.com/nbassler/bmod/blob/main/bmod_air.toml).

## Running the Analysis

### Step 1: Image Processing

```bash
PYTHONPATH=. python3 bmod/xrv_main.py -v /path/to/images -o output.csv
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
2. Perform a quadratic fit to extract Twiss parameters
3. Generate plots of the fits for each energy
4. Output `twiss_results_quadratic.csv`

Options:

| Option | Description | Default |
|--------|-------------|---------|
| `--sad VALUE` | Source-Axis Distance: distance from source to isocenter (IEC: z=0) in mm | `500.0` |
| `--z0 VALUE` | Fit anchor point in IEC nozzle coordinates (mm); see note below | `0.0` |
| `--cubic` | Also perform a cubic fit | off |
| `--no-plot` | Skip plot generation | off |
| `-a FILE` | Air-scan CSV/parquet; air scattering is subtracted before fitting | none |
| `-v` / `-vv` | Increase verbosity (info / debug) | warnings only |

#### Coordinate convention: IEC z vs. beam depth s

The fitting is performed in **beam-relative coordinates** s (Fermi-Eyges convention):

```
s = sad - z
```

where IEC z is the nozzle coordinate (z=0 at isocenter, positive toward the source):

| Position | IEC z | beam depth s |
|----------|-------|--------------|
| Source plane | z = +sad | s = 0 |
| Isocenter | z = 0 | s = sad |

`--z0` sets an **anchor point for the fit in IEC coordinates**. Internally it is converted to
`s₀ = sad − z0`. Its purpose is to help the fitting routine converge: choose a value that lies
well within the measured data range (i.e., surrounded by data points on both sides).
It is *not* required to coincide with any physically special position such as the isocenter.

## Output Interpretation

### Output Files

1. **Intermediate CSV**: Beam sizes at each position and energy
   - Columns: `z`, `s`, `energy`, `sigma_x_mm`, `sigma_y_mm`
   - `z` is the IEC nozzle coordinate; `s = sad - z` is the beam-relative depth

2. **Results CSVs**: One file per fit type
   - `twiss_results_quadratic.csv` — always produced
   - `twiss_results_cubic.csv` — produced when `--cubic` is passed
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

All fits are performed in beam-relative coordinates s (see [Coordinate convention](#coordinate-convention-iec-z-vs-beam-depth-s) above).

### Quadratic Model (Vacuum Propagation)

σ²(s) = a·(s−s₀)² + b·(s−s₀) + c

Represents ideal beam propagation in vacuum where:
- σ is the beam size
- s is the beam-relative depth (distance from source)
- s₀ is the fit anchor depth (converted from `--z0`: s₀ = sad − z0)

### Cubic Model (With Scattering)

σ²(s) = a·(s−s₀)² + b·(s−s₀) + c + d·(s−s₀)³

Extends the quadratic model to account for scattering effects in air, which become more significant at larger distances from the anchor point.


## Theoretical Background

The quadratic model is derived from the beam envelope equation in vacuum, where the beam size evolution is governed by the Courant-Snyder parameters:

σ²(s) = εβ(s) = ε(β₀ − 2α₀(s−s₀) + γ₀(s−s₀)²)

Where:
- ε is the emittance (a measure of beam quality)
- β₀, α₀, γ₀ are the Courant-Snyder parameters at the reference position

The cubic term accounts for multiple scattering effects in air, which cause the beam size to grow faster than quadratically with distance. This effect is more significant for:
- Lower energy beams
- Longer propagation distances through air


## License

[MIT License](LICENSE)
