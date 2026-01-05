# v1.0.0 - Initial Release: ML-based N₂O flux estimation in the Southern Ocean

## Overview

This is the first release of the `ml-argo-n2o` codebase, accompanying the preprint:

> Kelly, C.L., Chang, B.X., Emmanuelli, A., Park, E., Macdonald, A., & Nicholson, D.P. (2025). Low-pressure storms drive nitrous oxide emissions in the Southern Ocean. *Research Square* (Preprint). https://doi.org/10.21203/rs.3.rs-6378208/v1

## Features

This release provides a complete pipeline for:

- **Model Training** (`trainrf_v2.py`): Train four Random Forest models to predict nitrous oxide (N₂O) partial pressure from oceanographic variables (temperature, salinity, dissolved oxygen, nitrate)
- **BGC-Argo Application** (`applyrf_v2.py`): Apply trained models to BGC-Argo float profiles to generate pN₂O predictions with uncertainty estimates
- **Flux Calculations** (`flux_uncertainties_v3.py`): Calculate air-sea N₂O fluxes using predicted pN₂O with Monte Carlo uncertainty propagation
- **Visualization**: Generate maps, time series, and performance metrics for all main text figures

## Repository Contents

### Core Scripts
- `trainrf_v2.py` - Random Forest model training
- `applyrf_v2.py` - Model application to BGC-Argo data
- `plot_predictedn2o.py` - Map visualization of predictions
- `montecarloarrays.py` - Monte Carlo array generation
- `flux_uncertainties_v3.py` - Air-sea flux calculations with uncertainties

### Helper Functions
- `ml_feature_lists.py` - Feature set definitions
- `readgoshipdata.py` - GO-SHIP data loading and preprocessing
- `plottraintest.py` - Model performance visualization

### Notebooks
- `save_figure_data.ipynb` - Compile data for figures
- `main_text_figures.ipynb` - Generate main text figures

## Data Availability

Input datasets and analysis products are archived at Zenodo: https://doi.org/10.5281/zenodo.17904982

## System Requirements

- Python 3.8+
- Key dependencies: `scikit-learn`, `pandas`, `numpy`, `xarray`, `gsw`, `cartopy`
- See `environment.yml` for complete dependency list

## Installation

```bash
git clone https://github.com/ckelly314/ml-argo-n2o.git
cd ml-argo-n2o
conda env create -f environment.yml
conda activate ml-argo-n2o
```

## Quick Start

Run the complete pipeline:

```bash
python trainrf_v2.py          # Train models
python applyrf_v2.py          # Generate predictions
python plot_predictedn2o.py   # Visualize results
python flux_uncertainties_v3.py  # Calculate fluxes
```

Expected runtime: ~25 minutes on a 4-core desktop

## Citation

If you use this code in your research, please cite the paper:

> C.L. Kelly, B.X. Chang, A. Emmanuelli, E. Park, A. Macdonald, & D.P. Nicholson. Low-pressure storms drive nitrous oxide emissions in the Southern Ocean, 30 April 2025, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-6378208/v1].

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub or contact Colette Kelly (colette.kelly@whoi.edu).