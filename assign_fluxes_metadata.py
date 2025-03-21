"""
File: assign_metadata.py
---------------------------
Created on Thu Mar 20, 2025

Convert dataset of air-sea fluxes to netCDF, assign metadata,
and save out in .parquet, .csv., and .nc formats.

@author: Colette Kelly (colette.kelly@whoi.edu)
"""

import pandas as pd
import xarray as xr
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from initialize_paths import initialize_paths
import datetime as dt

# function to stitch together yearly datasets and assign metadata
def convert_fluxesparquet(t):
    data = t.to_pandas()
    data["PROFILE_ID"] = (data.PLATFORM_NUMBER.astype(str) + "_" + data.CYCLE_NUMBER.astype(str))
    data = data.set_index("PROFILE_ID")
    # drop duplicate variables
    del data["f"]
    del data["s"]
    
    return data

def saveoutfluxesnc(ds_augmented):
    ds_augmented.to_netcdf("datasets/fluxdataset.nc")

def saveoutfluxespq(ds_augmented, t):
    metadata = {
        "dataset_attrs": ds_augmented.attrs,
        "coordinate_attrs": {
            coord: ds_augmented[coord].attrs for coord in ds_augmented._coord_names
        },
        "variable_attrs": {
            var: ds_augmented[var].attrs for var in ds_augmented.data_vars
        },
    }
    table = t.replace_schema_metadata(
        {key: str(value) for key, value in metadata.items()}
    )

    pq.write_table(table, "datasets/fluxdataset.parquet")

def saveoutfluxescsv(data, ds_augmented):
    # Define metadata as commented lines
    metadata_lines = [
        f"# {ds_augmented.attrs['TITLE']}".replace(",", ""),
        "#",
        "# DATA CITATION",
        "# Please cite the paper when downloading data:",
        "# C.L. Kelly B.X. Chang A. Emmanuelli E. Park A. Macdonald & D.P. Nicholson (in prep).",
        "# Low-pressure storms drive nitrous oxide emissions in the Southern Ocean.",
        "# Please also acknowledge GO-SHIP and GO-BGC:",
        "# ML training data were collected and made publicly available by the U.S. Global Ship-based Hydrographic Investigations Program (U.S. GO-SHIP; https://usgoship.ucsd.edu/) and the programs that contribute to it.",
        "# BGC Argo data were assembled or collected and made available by the Global Ocean Biogeochemistry Array (GO-BGC) Project funded by National Science Foundation (NSF Award 1946578).",
        "#"]

    for item in ["CREATED", "MODIFIED", "AUTHOR"]:
        textstr = f"# {item} : {ds_augmented.attrs[item]}"
        metadata_lines.append(textstr)

    metadata_lines.append("#")

    units = {}
    for col in data.columns:
        if "units" in ds_augmented[col].attrs:
            units[col] = ds_augmented[col].attrs["units"].replace(",", "")
        else:
            units[col] = ""

    descriptions = {}
    for col in data.columns:
        if "description" in ds_augmented[col].attrs:
            descriptions[col] = ds_augmented[col].attrs["description"].replace(",", "")
        else:
            descriptions[col] = ""

    # Create a DataFrame with units as the second row
    data = pd.concat([pd.DataFrame([descriptions]), pd.DataFrame([units]), data], ignore_index=True)
    csv_filename = "datasets/fluxdataset.csv"
    with open(csv_filename, "w") as f:
        f.write("\n".join(metadata_lines) + "\n")
        data.to_csv(f, index=False)

def assign_fluxesmetadata(data):
    ds = data.to_xarray()
    
    ds = ds.assign_coords(
        LATITUDE=("PROFILE_ID", data["LATITUDE"]),
        LONGITUDE=("PROFILE_ID", data["LONGITUDE"]),
    )
    
    ds.attrs = {
        'TITLE': 'Southern Ocean Nitrous Oxide Predicted pN2O and Air-Sea Fluxes',
        'DESCRIPTION': 'pN2O predicted from BGC-Argo data with machine learning; fluxes calculated based on Wanninkhof et al. (2014) and Liang et al. (2013)',
        'CREATED': '2025-03-20 09:15:00',
        'MODIFIED': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'DATA CITATION': 'C.L. Kelly, B.X. Chang, A. Emmanuelli, E. Park, A. Macdonald, & D.P. Nicholson (in prep). Low-pressure storms drive nitrous oxide emissions in the Southern Ocean.',
        'GO-SHIP ACKNOWLEDGEMENTS':'ML training data were collected and made publicly available by the U.S. Global Ship-based Hydrographic Investigations Program (U.S. GO-SHIP; https://usgoship.ucsd.edu/) and the programs that contribute to it.',
        'BGC ARGO ACKNOWLEDGEMENTS': 'BGC Argo data were assembled or collected and made available by the Global Ocean Biogeochemistry Array (GO-BGC) Project funded by National Science Foundation (NSF Award 1946578).',
        'OTHER SOURCES':'All air-sea fluxes calculated with the gasex-python package https://github.com/boom-lab/gasex-python',
        'AUTHOR': 'Colette Kelly (colette.kelly@whoi.edu)'
        }
    
    argods = xr.open_dataset("datasets/argodataset.nc") # template attributes for CTD data
    
    for var in list(ds.coords):  # List of variables to copy attrs
        if var in argods and var in ds:
            ds[var].attrs = argods[var].attrs
    
    for var in list(ds.data_vars):  # List of variables to copy attrs
        if var in argods and var in ds:
            ds[var].attrs = argods[var].attrs
    
    ds["CT"].attrs = argods["Temp"].attrs
    ds["SA"].attrs = argods["Salinity"].attrs
    ds["zone"].attrs = {
            'long_name': 'Southern Ocean zone classification',
        'standard_name': 'southern_ocean_zone',
        'description': 'Assigned oceanic zone based on latitude and frontal boundaries from Gray et al. (2018)',
        'valid_values': ['STZ', 'SAZ', 'PFZ', 'ASZ', 'SIZ'],
        'meaning': 'STZ = Subtropical Zone, SAZ = Subantarctic Zone, PFZ = Polar Frontal Zone, ASZ = Antarctic Southern Zone, SIZ = Southern Ice Zone',
        'C_format': '%s',
        'FORTRAN_format': 'A3',
        'casted': 1
    }
    
    ds["month"].attrs = {
        'long_name': 'Month of observation',
        'standard_name': 'time_month',
        'description': 'Month of the year when the observation was recorded (1.0 = January, 12.0 = December)',
        'units': '1',  # Dimensionless, categorical integer stored as float
        'valid_min': 1.0,
        'valid_max': 12.0,
        'C_format': '%4.1f',
        'FORTRAN_format': 'F4.1',
        'casted': 1
    }
    
    # Assign attributes to N2Osol (N2O solubility)
    ds["N2Osol"].attrs = {
        'long_name': 'N2O Solubility',
        'description': 'Solubility of N2O in seawater',
        'units': 'mol L-1 atm-1',
        'valid_min': 0.014,
        'valid_max': 0.059,
        'source': 'Weiss and Price, 1980'
    }
    
    # Assign attributes to LON1
    ds["LON1"].attrs = {
        'long_name': 'Longitude Transform 1',
        'description': 'Cosine transformation of the longitude, recentered on 110 degrees east',
        'units': 'dimensionless',
        'valid_min': -1,
        'valid_max': 1,
        'source': 'Calculated using cos(pi * (LONGITUDE - 110) / 180)'
    }
    
    # Assign attributes to LON2
    ds["LON2"].attrs = {
        'long_name': 'Longitude Transform 2',
        'description': 'Cosine transformation of the longitude, recentered on 20 degrees east',
        'units': 'dimensionless',
        'valid_min': -1,
        'valid_max': 1,
        'source': 'Calculated using cos(pi * (LONGITUDE - 20) / 180)'
    }
    
    # Assign attributes to pN2O_pred
    ds["pN2O_pred"].attrs = {
        'long_name': 'Predicted seawater pN2O',
        'description': 'Partial pressure of N2O in seawater, mean of predictions by four random forest models',
        'units': 'natm',
        'valid_min': 290,
        'valid_max': 550
    }
    
    # Assign attributes to pN2O_pred
    ds["pN2O_predstd"].attrs = {
        'long_name': 'Seawater pN2O uncertainty',
        'description': 'Partial pressure of N2O in seawater, standard deviation of predictions by four random forest models',
        'units': 'natm',
        'valid_min': 0,
        'valid_max': 80,
    }
    
    # Assign attributes to N2O_nM
    ds["N2O_nM"].attrs = {
        'long_name': 'N2O Concentration in nanomolar',
        'description': 'Concentration of N2O in seawater, calculated from predicted pN2O and solubility',
        'units': 'nM',
        'valid_min': 6,
        'valid_max': 28,
        'source': 'Calculated as the product of pN2O_pred, N2Osol, and converted to nM'
    }
    
    # Assign attributes to C
    ds["C"].attrs = {
        'long_name': 'N2O Concentration in micromolar',
        'description': 'N2O concentration in micromoles per liter, converted from nM',
        'units': 'uM',
        'valid_min': 6e-6,
        'valid_max': 28e-6,
        'source': 'Calculated by converting N2O_nM from nM to uM'
    }
    
    ds["fugacityfactor"].attrs = {
        'long_name': 'Fugacity Factor',
        'description': 'Fugacity factor for N2O, such that pN2Oatm = XN2O*(SLP - vp_sw)*f',
        'units': 'dimensionless',
        'valid_min': 0.995,
        'valid_max': 0.997,
        'source': 'Weiss and Price, 1980'
    }
    
    ds["pN2Oatm"].attrs = {
        'long_name': 'Atmospheric pN2O',
        'description': 'Partial pressure of N2O in the atmosphere, pN2Oatm = XN2O*(SLP - vp_sw)*(fugacity factor)',
        'units': 'natm',
        'valid_min': np.min(ds["pN2Oatm"].values),
        'valid_max': np.max(ds["pN2Oatm"].values)
    }

    ds["DpN2O_pred"].attrs = {
        'long_name': 'Predicted N2O air-sea disequilibrium',
        'description': 'Difference between predicted seawater pN2O and atmospheric pN2O, pN2Osw - pN2Oatm',
        'units': 'natm',
        'valid_min': np.min(ds["DpN2O_pred"].values),
        'valid_max': np.max(ds["DpN2O_pred"].values),
        'source': 'pN2Osw - pN2Oatm'
    }
    
    ds["DpN2O_pred2"].attrs = {
        'long_name': 'Predicted N2O air-sea disequilibrium, assuming 1 atm',
        'description': 'Difference between predicted seawater pN2O and atmospheric pN2O, pN2Osw - pN2Oatm, where pN2Oatm is calculated assuming 1 atm barometric pressure',
        'units': 'natm',
        'valid_min': np.min(ds["DpN2O_pred2"].values),
        'valid_max': np.max(ds["DpN2O_pred2"].values)
    }
    
    ds["XN2Oa_sd"].attrs = {
        "long_name": "Atmospheric N2O Mole Fraction Uncertainty",
        "description": "Reported uncertainty for mole fraction of atmospheric N2O in air, derived from NOAA flask data",
        "units": "mol/mol",
        "valid_min": 3.20e-07,
        "valid_max": 3.38e-07
    }
    
    ds["mslcount"].attrs = {
        'long_name': 'Profiles per zone per month',
        'description': 'Number of profiles in the same zone and month as this data point',
        'units': '1',
        'valid_min': 318,
        'valid_max': 694,
    }
    
    ds["m2"] = ds["m2"].astype("int64")
    ds["m2"].attrs = {
        'long_name': 'Zone area',
        'description': 'Area of the zone in which profile occurred',
        'units': 'm2',
        'valid_min': 1.28e13,
        'valid_max': 2.26e13,
        'source': 'Gray et al., 2018'
    }
    
    ds["daysinmonth"] = ds["daysinmonth"].astype("int64")
    ds["daysinmonth"].attrs = {
        'long_name': 'Days in month',
        'description': 'Days of the month in which profile occurred',
        'units': 'days',
        'valid_min': 28,
        'valid_max': 31,
    }
    
    ds["ph2ov"].attrs = {
        'long_name': 'Vapor pressure of seawater, calculated with Gasex-Python toolbox',
        'units': 'atm',
        'valid_min': 0.005,
        'valid_max': 0.035,
        'source':'Gasex toolbox'
    }
    
    ds["FtW14"].attrs = {
        'long_name': 'Air-sea N2O flux',
        'description': 'Air-sea N2O flux calculated with Wanninkhof et al., 2014',
        'units': 'umol/m2/day',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW14stdev"].attrs = {
        'long_name': 'Air-sea N2O flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Wanninkhof et al., 2014',
        'units': 'umol/m2/day',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW14noice"].attrs = {
        'long_name': 'Air-sea N2O flux',
        'description': 'Air-sea N2O flux calculated with Wanninkhof et al., 2014 and no sea ice correction',
        'units': 'umol/m2/day',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW14stdevnoice"].attrs = {
        'long_name': 'Air-sea N2O flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Wanninkhof et al., 2014 and no sea ice correction',
        'units': 'umol/m2/day',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW141atm"].attrs = {
        'long_name': 'Air-sea N2O flux',
        'description': 'Air-sea N2O flux calculated with Wanninkhof et al., 2014 and 1 atm barometric pressure',
        'units': 'umol/m2/day',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW14stdev1atm"].attrs = {
        'long_name': 'Air-sea N2O flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Wanninkhof et al., 2014 and 1 atm barometric pressure',
        'units': 'umol/m2/day',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW14medN2O"].attrs = {
        'long_name': 'Air-sea N2O flux',
        'description': 'Air-sea N2O flux calculated with Wanninkhof et al., 2014 and the median Southern Ocean pN2O',
        'units': 'umol/m2/day',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW14stdevmedN2O"].attrs = {
        'long_name': 'Air-sea N2O flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Wanninkhof et al., 2014 and the median Southern Ocean pN2O',
        'units': 'umol/m2/day',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW14WINDS"].attrs = {
        'long_name': 'Air-sea N2O flux',
        'description': 'Air-sea N2O flux calculated with Wanninkhof et al., 2014 and 25% higher wind speeds',
        'units': 'umol/m2/day',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW14stdevWINDS"].attrs = {
        'long_name': 'Air-sea N2O flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Wanninkhof et al., 2014 and 25% higher wind speeds',
        'units': 'umol/m2/day',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW14COMBINED"].attrs = {
        'long_name': 'Air-sea N2O flux',
        'description': 'Air-sea N2O flux calculated with Wanninkhof et al., 2014 and no ice correction and 25% higher wind speeds',
        'units': 'umol/m2/day',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW14stdevCOMBINED"].attrs = {
        'long_name': 'Air-sea N2O flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Wanninkhof et al., 2014 and no ice correction and 25% higher wind speeds',
        'units': 'umol/m2/day',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW14CYCLONES"].attrs = {
        'long_name': 'Air-sea N2O flux',
        'description': 'Air-sea N2O flux calculated with Wanninkhof et al., 2014 and 0.01 atm lower barometric pressure',
        'units': 'umol/m2/day',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW14stdevCYCLONES"].attrs = {
        'long_name': 'Air-sea N2O flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Wanninkhof et al., 2014 and 0.01 atm lower barometric pressure',
        'units': 'umol/m2/day',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FdL13"].attrs = {
        'long_name': 'Diffusive component of air-sea N2O flux',
        'description': 'Diffusive component of air-sea N2O flux calculated with Liang et al., 2013',
        'units': 'umol/m2/day',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al., 2013'
    }
    
    ds["FcL13"].attrs = {
        'long_name': 'Small bubble component of air-sea N2O flux',
        'description': 'Small bubble component of air-sea N2O flux calculated with Liang et al., 2013',
        'units': 'umol/m2/day',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al., 2013'
    }
    
    ds["FpL13"].attrs = {
        'long_name': 'Large bubble component of air-sea N2O flux',
        'description': 'Large bubble component of air-sea N2O flux calculated with Liang et al., 2013',
        'units': 'umol/m2/day',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al., 2013'
    }
    
    ds["FtL13"].attrs = {
        'long_name': 'Air-sea N2O flux',
        'description': 'Air-sea N2O flux calculated with Liang et al., 2013',
        'units': 'umol/m2/day',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al., 2013'
    }
    
    ds["FdL13stdev"].attrs = {
        'long_name': 'Diffusive component of air-sea N2O flux uncertainty',
        'description': 'Uncertainty in diffusive component of air-sea N2O flux calculated with Liang et al., 2013',
        'units': 'umol/m2/day',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al., 2013'
    }
    
    ds["FcL13stdev"].attrs = {
        'long_name': 'Small bubble component of air-sea N2O flux uncertainty',
        'description': 'Uncertainty in small bubble component of air-sea N2O flux calculated with Liang et al., 2013',
        'units': 'umol/m2/day',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al., 2013'
    }
    
    ds["FpL13stdev"].attrs = {
        'long_name': 'Large bubble component of air-sea N2O flux uncertainty',
        'description': 'Uncertainty in large bubble component of air-sea N2O flux calculated with Liang et al., 2013',
        'units': 'umol/m2/day',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al., 2013'
    }
    
    ds["FtL13stdev"].attrs = {
        'long_name': 'Air-sea N2O flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Liang et al., 2013',
        'units': 'umol/m2/day',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al., 2013'
    }
    
    ds["FtL13noice"].attrs = {
        'long_name': 'Air-sea N2O flux',
        'description': 'Air-sea N2O flux calculated with Liang et al., 2013 and no sea ice correction',
        'units': 'umol/m2/day',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al., 2013'
    }
    
    ds["FtL13stdevnoice"].attrs = {
        'long_name': 'Air-sea N2O flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Liang et al., 2013 and no sea ice correction',
        'units': 'umol/m2/day',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Liang et al., 2013'
    }
    
    ds["FtL131atm"].attrs = {
        'long_name': 'Air-sea N2O flux',
        'description': 'Air-sea N2O flux calculated with Liang et al., 2013and 1 atm barometric pressure',
        'units': 'umol/m2/day',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al., 2013'
    }
    
    ds["FtL13stdev1atm"].attrs = {
        'long_name': 'Air-sea N2O flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Liang et al., 2013 and 1 atm barometric pressure',
        'units': 'umol/m2/day',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Liang et al., 2013'
    }
    
    ds["FtL13medN2O"].attrs = {
        'long_name': 'Air-sea N2O flux',
        'description': 'Air-sea N2O flux calculated with Liang et al., 2013 and the median Southern Ocean pN2O',
        'units': 'umol/m2/day',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al., 2013'
    }
    
    ds["FtL13stdevmedN2O"].attrs = {
        'long_name': 'Air-sea N2O flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Liang et al., 2013 and the median Southern Ocean pN2O',
        'units': 'umol/m2/day',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Liang et al., 2013'
    }

    ds["FtL13medK"].attrs = {
        'long_name': 'Air-sea N2O flux',
        'description': 'Air-sea N2O flux calculated with Liang et al., 2013 and median piston velocities',
        'units': 'umol/m2/day',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al., 2013'
    }
    
    ds["FtL13stdevmedK"].attrs = {
        'long_name': 'Air-sea N2O flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Liang et al., 2013 and median piston velocities',
        'units': 'umol/m2/day',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Liang et al., 2013'
    }
    
    ds["FtL13WINDS"].attrs = {
        'long_name': 'Air-sea N2O flux',
        'description': 'Air-sea N2O flux calculated with Liang et al., 2013and 25% higher wind speeds',
        'units': 'umol/m2/day',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al., 2013'
    }
    
    ds["FtL13stdevWINDS"].attrs = {
        'long_name': 'Air-sea N2O flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Liang et al., 2013 and 25% higher wind speeds',
        'units': 'umol/m2/day',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Liang et al., 2013'
    }
    
    ds["FtL13COMBINED"].attrs = {
        'long_name': 'Air-sea N2O flux',
        'description': 'Air-sea N2O flux calculated with Liang et al., 2013 and no ice correction and 25% higher wind speeds',
        'units': 'umol/m2/day',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al., 2013'
    }
    
    ds["FtL13stdevCOMBINED"].attrs = {
        'long_name': 'Air-sea N2O flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Liang et al., 2013 and no ice correction and 25% higher wind speeds',
        'units': 'umol/m2/day',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Liang et al., 2013'
    }
    
    ds["FtL13CYCLONES"].attrs = {
        'long_name': 'Air-sea N2O flux',
        'description': 'Air-sea N2O flux calculated with Liang et al., 2013 and 0.01 atm lower barometric pressure',
        'units': 'umol/m2/day',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al., 2013'
    }
    
    ds["FtL13stdevCYCLONES"].attrs = {
        'long_name': 'Air-sea N2O flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Liang et al., 2013 and 0.01 atm lower barometric pressure',
        'units': 'umol/m2/day',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Liang et al., 2013'
    }
    
    ds["Ft"].attrs = {
        'long_name': 'Air-sea N2O flux',
        'description': 'Air-sea N2O flux',
        'units': 'umol/m2/day',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al., (2013), Wanninkhof et al. (2014)'
    }
    
    ds["Ftstdev"].attrs = {
        'long_name': 'Air-sea N2O flux uncertainty',
        'description': 'Air-sea N2O flux uncertainty',
        'units': 'umol/m2/day',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al., (2013), Wanninkhof et al. (2014)'
    }

    ds["FtW14integrated"].attrs = {
        'long_name': 'Space- and time-integrated flux',
        'description': 'Space- and time-integrated flux calculated with Wanninkhof et al., 2014',
        'units': 'Tg N/year',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW14integratedstdev"].attrs = {
        'long_name': 'Space- and time-integrated flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Wanninkhof et al., 2014',
        'units': 'Tg N/year',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Wanninkhof et al. (2014)'
    }

    ds["FtW14integratednoice"].attrs = {
        'long_name': 'Space- and time-integrated flux',
        'description': 'Space- and time-integrated flux calculated with Wanninkhof et al., 2014 and no sea ice correction',
        'units': 'Tg N/year',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW14integratedstdevnoice"].attrs = {
        'long_name': 'Space- and time-integrated flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Wanninkhof et al., 2014 and no sea ice correction',
        'units': 'Tg N/year',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW14integrated1atm"].attrs = {
        'long_name': 'Space- and time-integrated flux',
        'description': 'Space- and time-integrated flux calculated with Wanninkhof et al., 2014 and 1 atm barometric pressure',
        'units': 'Tg N/year',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW14integratedstdev1atm"].attrs = {
        'long_name': 'Space- and time-integrated flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Wanninkhof et al., 2014 and 1 atm barometric pressure',
        'units': 'Tg N/year',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW14integratedmedN2O"].attrs = {
        'long_name': 'Space- and time-integrated flux',
        'description': 'Space- and time-integrated flux calculated with Wanninkhof et al., 2014 and the median Southern Ocean pN2O',
        'units': 'Tg N/year',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW14integratedstdevmedN2O"].attrs = {
        'long_name': 'Space- and time-integrated flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Wanninkhof et al., 2014 and the median Southern Ocean pN2O',
        'units': 'Tg N/year',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW14integratedWINDS"].attrs = {
        'long_name': 'Space- and time-integrated flux',
        'description': 'Space- and time-integrated flux calculated with Wanninkhof et al., 2014 and 25% higher wind speeds',
        'units': 'Tg N/year',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW14integratedstdevWINDS"].attrs = {
        'long_name': 'Space- and time-integrated flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Wanninkhof et al., 2014 and 25% higher wind speeds',
        'units': 'Tg N/year',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW14integratedCOMBINED"].attrs = {
        'long_name': 'Space- and time-integrated flux',
        'description': 'Space- and time-integrated flux calculated with Wanninkhof et al., 2014 and no ice correction and 25% higher wind speeds',
        'units': 'Tg N/year',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW14integratedstdevCOMBINED"].attrs = {
        'long_name': 'Space- and time-integrated flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Wanninkhof et al., 2014 and no ice correction and 25% higher wind speeds',
        'units': 'Tg N/year',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW14integratedCYCLONES"].attrs = {
        'long_name': 'Space- and time-integrated flux',
        'description': 'Space- and time-integrated flux calculated with Wanninkhof et al., 2014 and 0.01 atm lower barometric pressure',
        'units': 'Tg N/year',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Wanninkhof et al. (2014)'
    }
    
    ds["FtW14integratedstdevCYCLONES"].attrs = {
        'long_name': 'Space- and time-integrated flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Wanninkhof et al., 2014 and 0.01 atm lower barometric pressure',
        'units': 'Tg N/year',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Wanninkhof et al. (2014)'
    }

    ds["FtL13integrated"].attrs = {
        'long_name': 'Space- and time-integrated flux',
        'description': 'Space- and time-integrated flux calculated with Liang et al., 2012',
        'units': 'Tg N/year',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al. (2013)'
    }
    
    ds["FtL13integratedstdev"].attrs = {
        'long_name': 'Space- and time-integrated flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Liang et al., 2013',
        'units': 'Tg N/year',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Liang et al. (2013)'
    }
    
    ds["FtL13integratednoice"].attrs = {
        'long_name': 'Space- and time-integrated flux',
        'description': 'Space- and time-integrated flux calculated with Liang et al., 2013 and no sea ice correction',
        'units': 'Tg N/year',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al. (2013)'
    }
    
    ds["FtL13integratedstdevnoice"].attrs = {
        'long_name': 'Space- and time-integrated flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Liang et al., 2013 and no sea ice correction',
        'units': 'Tg N/year',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Liang et al. (2013)'
    }
    
    ds["FtL13integrated1atm"].attrs = {
        'long_name': 'Space- and time-integrated flux',
        'description': 'Space- and time-integrated flux calculated with Liang et al., 2013 and 1 atm barometric pressure',
        'units': 'Tg N/year',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al. (2013)'
    }
    
    ds["FtL13integratedstdev1atm"].attrs = {
        'long_name': 'Space- and time-integrated flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Liang et al., 2013 and 1 atm barometric pressure',
        'units': 'Tg N/year',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Liang et al. (2013)'
    }
    
    ds["FtL13integratedmedN2O"].attrs = {
        'long_name': 'Space- and time-integrated flux',
        'description': 'Space- and time-integrated flux calculated with Liang et al., 2013 and the median Southern Ocean pN2O',
        'units': 'Tg N/year',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al. (2013)'
    }
    
    ds["FtL13integratedstdevmedN2O"].attrs = {
        'long_name': 'Space- and time-integrated flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Liang et al., 2013 and the median Southern Ocean pN2O',
        'units': 'Tg N/year',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Liang et al. (2013)'
    }

    ds["FtL13integratedmedK"].attrs = {
        'long_name': 'Space- and time-integrated flux',
        'description': 'Space- and time-integrated flux calculated with Liang et al., 2013 and median piston velocities',
        'units': 'Tg N/year',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al. (2013)'
    }
    
    ds["FtL13integratedstdevmedK"].attrs = {
        'long_name': 'Space- and time-integrated flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Liang et al., 2013 and median piston velocities',
        'units': 'Tg N/year',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Liang et al. (2013)'
    }
    
    ds["FtL13integratedWINDS"].attrs = {
        'long_name': 'Space- and time-integrated flux',
        'description': 'Space- and time-integrated flux calculated with Liang et al., 2013 and 25% higher wind speeds',
        'units': 'Tg N/year',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al. (2013)'
    }
    
    ds["FtL13integratedstdevWINDS"].attrs = {
        'long_name': 'Space- and time-integrated flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Liang et al., 2013 and 25% higher wind speeds',
        'units': 'Tg N/year',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Liang et al. (2013)'
    }
    
    ds["FtL13integratedCOMBINED"].attrs = {
        'long_name': 'Space- and time-integrated flux',
        'description': 'Space- and time-integrated flux calculated with Liang et al., 2013 and no ice correction and 25% higher wind speeds',
        'units': 'Tg N/year',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al. (2013)'
    }
    
    ds["FtL13integratedstdevCOMBINED"].attrs = {
        'long_name': 'Space- and time-integrated flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Liang et al., 2013 and no ice correction and 25% higher wind speeds',
        'units': 'Tg N/year',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Liang et al. (2013)'
    }
    
    ds["FtL13integratedCYCLONES"].attrs = {
        'long_name': 'Space- and time-integrated flux',
        'description': 'Space- and time-integrated flux calculated with Liang et al., 2013 and 0.01 atm lower barometric pressure',
        'units': 'Tg N/year',
        'valid_min': -10,
        'valid_max': 50,
        'reference': 'Liang et al. (2013)'
    }
    
    ds["FtL13integratedstdevCYCLONES"].attrs = {
        'long_name': 'Space- and time-integrated flux uncertainty',
        'description': 'Uncertainty in air-sea N2O flux calculated with Liang et al., 2013 and 0.01 atm lower barometric pressure',
        'units': 'Tg N/year',
        'valid_min': 0,
        'valid_max': 40,
        'reference': 'Liang et al. (2013)'
    }

    return ds

def main():
    datapath, argopath, outputpath, era5path = initialize_paths()
    t = pq.read_table(f"{outputpath}/fluxes.parquet")
    data = convert_fluxesparquet(t)
    ds_augmented = assign_fluxesmetadata(data)
    saveoutfluxesnc(ds_augmented)
    saveoutfluxespq(ds_augmented, t)
    saveoutfluxescsv(data, ds_augmented)


if __name__ == "__main__":
    main()
