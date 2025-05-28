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
    df["N2Osol"] = sol.sol_SP_pt(df.PSAL_ADJUSTED,df.pt,gas="N2O",slp = df.msl, units="M")

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
    
    # take mean and stdev of pN2O predicted by each model
    n2opred = np.mean(predictions, axis=1)
    n2opredstd = np.std(predictions, axis=1)

    # store values
    df["pN2O_pred"] = n2opred
    df["pN2O_predstd"] = n2opredstd
    df["N2O_nM"] = df["pN2O_pred"]*df["N2Osol"]
    df["C"] = df["N2O_nM"]*1e-6

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
    mslarray = np.array(df.msl)
    pN2Osw = np.array(df.pN2O_pred)
    n2o_atm_ppb = np.array(df['n2o_atm'])
    
    # calculate N2O fugacity
    f = fugacity_factor(ptarray,gas="N2O",slp=mslarray)
    
    # calculate atmospheric partial pressure of N2O and N2O disequilibrium
    rh = 1 # can replace with calculated relative humidity but for now assume rh=100% at moist interface
    ph2oveq = vpress_sw(SParray,ptarray)
    ph2ov = rh * ph2oveq
    n2o_atm_natm = xN2Oarray * (mslarray - ph2ov) * f 
    DpN2O = pN2Osw - n2o_atm_natm
    
    df["fugacityfactor"] = f
    df["pN2Oatm"] = n2o_atm_natm
    df["DpN2O_pred"] = DpN2O
    df["DpN2O_pred2"] = pN2Osw - xN2Oarray

    return df

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

    # save out
    output.to_parquet("datasets/n2opredictions.parquet",
        engine='pyarrow',
        coerce_timestamps='us') # convert all timestamps to microsecond precision
    print(f"predicted N2O saved out to datasets/n2opredictions.parquet")

if __name__ == "__main__":
    main()
