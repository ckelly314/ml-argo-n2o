"""
File: applyrf_v2.py
-----------------------------
Created on Weds March 12, 2025

Python script to read in BGC-Argo float data and
apply four core Random Forest models to predict
float-based pN2O.

@author: Colette Kelly (colette.kelly@whoi.edu)
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# helper functions
from ml_feature_lists import feature_lists

# ML functions
from joblib import load

# functions from gasex module
import gasex.sol as sol
from gasex.phys import vpress_sw
from gasex.fugacity import fugacity_factor
from gasex.sol import sol_SP_pt

def load_argo_data(path_to_data):
    """
    Load Argo parquet files containing interpolated ERA5 wind speed, 
    sea level pressure, and sea ice data. 
    
    Parameters:
    path_to_data (str): Path to the parquet file containing Argo data.
    
    Returns:
    pandas.DataFrame: Argo data with additional calculated fields:
        - 'N2Osol': N2O solubility calculated using gasex module.
        - 'LON1', 'LON2': Longitude transformations.
    """
    t = pq.read_table(path_to_data)
    # convert to pandas dataframe
    df = t.to_pandas()
    print(f"argo data read in from {path_to_data}")

    # calculate N2O solubility with gasex module
    df["N2Osol_era5"] = sol.sol_SP_pt(df.PSAL_ADJUSTED,df.pt,gas="N2O",slp = df.msl_era5, units="M")
    df["N2Osol_ncep"] = sol.sol_SP_pt(df.PSAL_ADJUSTED,df.pt,gas="N2O",slp = df.msl_ncep, units="M")

    # calculate longitude transforms
    df["LON1"] = np.cos(np.pi*(df["LONGITUDE"]-110)/180)
    df["LON2"] = np.cos(np.pi*(df["LONGITUDE"]-20)/180)

    return df

def generate_predictions(df, argo_feature_sets, modelIDs=[1,2,3,4]):
    """
    Generate pN2O predictions using pre-trained Random Forest models.
    
    Parameters:
    df (pandas.DataFrame): Dataframe containing Argo observations.
    argo_feature_sets (dict): Dictionary mapping model IDs to feature sets.
    modelIDs (list of int, optional): List of model IDs to use for predictions. Defaults to [1,2,3,4].
    
    Returns:
    pandas.DataFrame: Updated dataframe with additional columns:
        - 'pN2O_pred': Mean predicted pN2O values.
        - 'pN2O_predstd': Standard deviation of predicted pN2O values.
        - 'N2O_nM': N2O concentration in nanomolar.
        - 'C': N2O concentration in molar units.
    """
    predictions = np.ones((df.shape[0],len(modelIDs)))
    residual_errors = np.ones((df.shape[0],len(modelIDs)))
    tree_errors = np.ones((df.shape[0],len(modelIDs)))
    training_metrics = pd.read_csv(f'model_metrics.csv')

    # loop through models of choice
    for count, modelID in enumerate(modelIDs):
        print(f'loading model{modelID}_rf_full.joblib')
        
        # set up array of predictors
        feature_list = argo_feature_sets[modelID]
        predictors = np.array(df[feature_list])
        print(f"generating predictions from {feature_list}")

        # load trained model
        RF = load(f'model{modelID}_rf_full.joblib')

        # generate predictions
        predictions[:,count] = RF.predict(np.array(predictors))

        # Get predictions from all individual trees
        tree_predictions = np.array([tree.predict(predictors) for tree in RF.estimators_])
        tree_errors[:,count] = np.std(tree_predictions, axis=0)

        # calculate residual errors based on percent error
        # residual_errors[:,count] = predictions[:,count]*training_metrics['mpe'].iloc[count]
        # calculate residual errors based on mae
        residual_errors[:,count] = training_metrics['mae'].iloc[count]

        # take mean and stdev of pN2O predicted by each model
    n2opred = np.mean(predictions, axis=1)
    model_uncertainty = np.std(predictions, axis=1) # model disagreement
    total_uncertainty = np.sqrt(model_uncertainty**2 +
                            np.sum(tree_errors**2*(1/len(modelIDs))**2, axis = 1) + #  variance across different trees in the forest
                            np.sum(residual_errors**2*(1/len(modelIDs))**2, axis = 1)) # inherent noise in the data

    # store values
    df["pN2O_pred"] = n2opred
    df["pN2O_predstd"] = total_uncertainty

    df["N2O_nM_era5"] = df["pN2O_pred"]*df["N2Osol_era5"]
    df["C_era5"] = df["N2O_nM_era5"]*1e-6

    df["N2O_nM_ncep"] = df["pN2O_pred"]*df["N2Osol_ncep"]
    df["C_ncep"] = df["N2O_nM_ncep"]*1e-6
    print(f"median predicted surface pN2O = {np.median(df.pN2O_pred):.4}+/-{np.std(df.pN2O_pred):.3}")

    return df

def calculate_pN2Oatm(df):
    """
    Calculate atmospheric partial pressure of N2O and N2O disequilibrium.
    
    Parameters:
    df (pandas.DataFrame): Dataframe containing Argo observations with pN2O predictions.
    
    Returns:
    pandas.DataFrame: Updated dataframe with additional columns:
        - 'fugacityfactor': Fugacity factor for N2O.
        - 'pN2Oatm': Atmospheric partial pressure of N2O.
        - 'DpN2O_pred': N2O disequilibrium.
        - 'DpN2O_pred2': Alternative calculation of N2O disequilibrium.
    """
    df = df.rename(columns = {'Temp':'CT',
                          'Salinity':'SA'})

    # calculate derived variables needed for air-sea flux calculation
    xN2Oarray = np.array(df["XN2Oa"]*1e9)
    ptarray = np.array(df.pt)
    SParray = np.array(df.SP)
    mslarray_era5 = np.array(df.msl_era5)
    mslarray_ncep = np.array(df.msl_ncep)
    pN2Osw = np.array(df.pN2O_pred)
    n2o_atm_ppb = np.array(df['n2o_atm'])
    
    # calculate N2O fugacity
    f_era5 = fugacity_factor(ptarray,gas="N2O",slp=mslarray_era5)
    f_ncep = fugacity_factor(ptarray,gas="N2O",slp=mslarray_ncep)
    
    # calculate atmospheric partial pressure of N2O and N2O disequilibrium
    rh = 1 # can replace with calculated relative humidity but for now assume rh=100% at moist interface
    ph2oveq = vpress_sw(SParray,ptarray)
    ph2ov = rh * ph2oveq
    n2o_atm_natm_era5 = xN2Oarray * (mslarray_era5 - ph2ov) * f_era5
    n2o_atm_natm_ncep = xN2Oarray * (mslarray_ncep - ph2ov) * f_ncep
    DpN2O_era5 = pN2Osw - n2o_atm_natm_era5
    DpN2O_ncep = pN2Osw - n2o_atm_natm_ncep
    
    df["fugacityfactor_era5"] = f_era5
    df["fugacityfactor_ncep"] = f_ncep
    df["pN2Oatm_era5"] = n2o_atm_natm_era5
    df["pN2Oatm_era5"] = n2o_atm_natm_ncep
    df["DpN2O_pred_era5"] = DpN2O_era5
    df["DpN2O_pred_ncep"] = DpN2O_ncep
    df["DpN2O_pred2"] = pN2Osw - xN2Oarray

    return df

def calc_fluxvars(data):
    # calculate area-time per profile for integrating fluxes in space and time
    data["month"] = data.JULD.dt.month # we'll use this to group data by zone and month
    # source for zone areas: Gray et al. (2018)
    # set up dataframes containing zone areas and days in month to calculate integrated fluxes
    areas = pd.DataFrame([["STZ",2.26e7],["SAZ",1.94e7],["PFZ",1.43e7],
                           ["ASZ",1.28e7],["SIZ",1.72e7],["TOTAL",8.64e7]],
                columns = ["zone","Area_km2"]).set_index("zone")
    
    areas["m2"] = areas.Area_km2*1e6 # convert to m2 because fluxes are in umol/m2/day
    
    daysinmonth = pd.DataFrame([[1.0, 31],
                                [2.0,28],
                                [3.0,31],
                                [4.0,30],
                                [5.0,31],
                                [6.0,30],
                                [7.0, 31],
                                [8.0,31],
                                [9.0,30],
                                [10.0,31],
                                [11.0,30],
                                [12.0,31],
                               ],
                               columns = ["month","daysinmonth"]).set_index("month")
    
    # need area and time covered by each float, per zone, per month
    counts = data[["zone", "month","msl_era5"]].groupby(["zone", "month"]).count() # how many floats in each zone and month?
    surface = data.set_index(["zone", "month"]).join(counts, rsuffix = "count").reset_index() # attach counts to df
    surface = surface.set_index("zone").join(areas["m2"]).reset_index() # attach zone areas to df
    surface = surface.set_index("month").join(daysinmonth["daysinmonth"]).reset_index() # attach days in month to df
    
    # other parameters needed for flux calculation
    surface["XN2Oa_sd"] = surface['n2o_atm_sd']*1e-9 # this gets used in the Monte Carlo analysis
    surface["ph2ov"] = vpress_sw(surface.SP,surface.pt) # atm
    surface["f_era5"]  = fugacity_factor(surface.pt,gas='N2O',slp=surface.msl_era5)
    surface["f_ncep"]  = fugacity_factor(surface.pt,gas='N2O',slp=surface.msl_ncep)
    surface["s"] = sol_SP_pt(surface.SP,surface.pt,chi_atm=surface.XN2Oa, gas='N2O',units="mM")
    surface["pN2Oatm_era5"] = surface.XN2Oa * surface.f_era5 * (surface.msl_era5 - surface.ph2ov)
    surface["pN2Oatm_ncep"] = surface.XN2Oa * surface.f_ncep * (surface.msl_ncep - surface.ph2ov)

    # check for NaN's to mask or drop
    vars_with_NaNs = []
    fluxvars = ["SP", "pt",
                "XN2Oa", "XN2Oa_sd",
                "pN2O_pred","C_era5", "C_ncep",
                "pN2Oatm_era5", "pN2Oatm_ncep",
                "U10_era5", "msl_era5", "SI_era5",
                "U10_ncep", "msl_ncep", "SI_ncep"
               ]    
    for var in fluxvars:
        if len(surface) != len(surface.dropna(subset=[var])):
            print(f"{var} contains NaNs")
            vars_with_NaNs.append(var)
    
    return surface

def main():
    """
    Main script execution function.
    Loads Argo data, applies trained Random Forest models to generate N2O predictions,
    calculates atmospheric partial pressure of N2O and disequilibrium, and saves results.
    """
    feature_sets, feature_set_labels, argo_feature_sets = feature_lists()

    # load argo surface data from parquet files
    argodata = load_argo_data("datasets/argodataset.parquet")

    # apply selected models to argo data
    n2opredictions = generate_predictions(argodata, argo_feature_sets, modelIDs=[1,2,3,4])

    # calculate atmospheric partial pressure of N2O and N2O disequilibrium
    output = calculate_pN2Oatm(n2opredictions)

    # convert JULD from ns to us
    for col in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[col]):
            print(f"Converting datetime column: {col}")
            output[col] = output[col].dt.floor('us')

    # other parameters needed for flux calculations
    output = calc_fluxvars(output)
    
    # save out
    output.to_parquet("datasets/n2opredictions.parquet") # convert all timestamps to microsecond precision
    print(f"predicted N2O saved out to datasets/n2opredictions.parquet")

if __name__ == "__main__":
    main()
