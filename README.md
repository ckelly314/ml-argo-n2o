# ml-argo-n2o

This repository contains code to train machine learning models for predicting nitrous oxide (N2O) from oceanographic variables and apply those models to Biogeochemical Argo (BGC-Argo) float data. The trained models are then used to estimate air-sea N2O fluxes and their uncertainties.

## Repository Contents

### Train Best Models

#### `trainrf_v2.py`
Train four core Random Forest models on the full training dataset (training + validation data) to predict N2O from the following oceanographic variables:
- Temperature (T)
- Salinity (S)
- Dissolved oxygen (O2)
- Nitrate (NO3⁻)

### Apply RF Models to Float Data

#### `applyrf_v2.py`
Read in BGC-Argo float data and apply the trained Random Forest models to predict partial pressure of N2O (`pN2O`) in the ocean.

#### `plot_predictedn2o.py`
Generate visualizations of the predicted N2O values, including:
- Maps of predicted N2O and associated uncertainties
- Histogram of uncertainty distributions

### Calculate Air-Sea Fluxes

#### `flux_uncertainties.py`
Compute air-sea N2O fluxes, incorporating Monte Carlo simulations to estimate uncertainties due to errors in:
- Predicted pN2O in seawater (`pN2Osw`)
- Atmospheric N2O mixing ratio (`XN2Oatm`)

#### `assign_fluxes_metadata.py`
Convert the computed air-sea N2O flux dataset into multiple formats with appropriate metadata:
- **NetCDF (`.nc`)** for compatibility with scientific workflows
- **Parquet (`.parquet`)** for efficient data storage and retrieval
- **CSV (`.csv`)** for general accessibility

## Usage
1. **Train the models**: Run `trainrf_v2.py` to train the Random Forest models.
2. **Apply to float data**: Use `applyrf_v2.py` to generate predicted N2O values from BGC-Argo profiles.
3. **Visualize predictions**: Execute `plot_predictedn2o.py` to create maps and uncertainty histograms.
4. **Calculate air-sea fluxes**: Run `flux_uncertainties.py` to estimate fluxes and their uncertainties.
5. **Save with metadata**: Use `assign_fluxes_metadata.py` to store results in different formats.

## Dependencies
- Python 3.x
- `scikit-learn`
- `numpy`
- `pandas`
- `xarray`
- `matplotlib`
- `cartopy`

## License
This project is licensed under [MIT License](LICENSE).

## Citation
If you use this code in your research, please cite the paper:
C.L. Kelly B.X. Chang A. Emmanuelli E. Park A. Macdonald & D.P. Nicholson (in prep). Low-pressure storms drive nitrous oxide emissions in the Southern Ocean.

## Contact
For questions or collaborations, please contact Colette Kelly (https://github.com/ckelly314).
