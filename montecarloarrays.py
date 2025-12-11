#!/usr/bin/env python3
"""
Set up and save out input arrays for Monte Carlo calculations.

Created Mon Jun 23 2025
@author: Colette Kelly
colette.kelly@whoi.edu
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def load_data():
    # base case
    t = pq.read_table(f"datasets/n2opredictions.parquet")
    df = t.to_pandas()

    return df

def genmontecarlo(iters=None, surface=None, outputpath = 'datasets'):
    rows = len(surface)
    ones = np.ones((rows,iters))
    # variables containing error
    pN2O_pred_reshaped = np.array(surface.pN2O_pred).reshape(-1,1)
    pN2O_sd_reshaped = np.array(surface.pN2O_predstd).reshape(-1,1)
    pN2Oerror = np.random.normal(loc=pN2O_pred_reshaped, scale=pN2O_sd_reshaped, size=(rows, iters))*1e-9
    np.save(f'{outputpath}/pN2Oerror.npy', pN2Oerror)
    # to re-load: pN2Oerror= np.load(f'{outputpath}/pN2Oerror.npy')

    for count, i in enumerate(range(-10,12,2)):
        df_trainbias = df_list[count]
        pN2O_pred_reshaped = np.array(df_trainbias.pN2O_pred).reshape(-1,1)
        pN2O_sd_reshaped = np.array(df_trainbias.pN2O_predstd).reshape(-1,1)
        pN2Oerror = np.random.normal(loc=pN2O_pred_reshaped, scale=pN2O_sd_reshaped, size=(rows, iters))*1e-9
        print(f'saving {outputpath}/pN2Oerror_trainbias{i}.npy')
        np.save(f'{outputpath}/pN2Oerror_trainbias{i}.npy', pN2Oerror)
    
    XN2O_reshaped = np.array(surface.XN2Oa).reshape(-1,1)
    XN2O_sd_reshaped = np.array(surface.XN2Oa_sd).reshape(-1,1)
    XN2Oerror = np.random.normal(loc=XN2O_reshaped, scale=XN2O_sd_reshaped, size=(rows, iters))
    np.save(f'{outputpath}/XN2Oerror.npy', XN2Oerror)
    
    # variables with no error
    SPerror = np.array(surface.SP).reshape(-1,1) * ones
    np.save(f'{outputpath}/SPerror.npy', SPerror)
    PTerror = np.array(surface.pt).reshape(-1,1) * ones
    np.save(f'{outputpath}/PTerror.npy', PTerror)

    U10_era5error = np.array([surface.U10_era5]).T * ones
    np.save(f'{outputpath}/U10_era5error.npy', U10_era5error)
    msl_era5error = np.array([surface.msl_era5]).T * ones
    np.save(f'{outputpath}/msl_era5error.npy', msl_era5error)
    SI_era5error = np.array(surface.SI_era5).reshape(-1,1) * ones
    np.save(f'{outputpath}/SI_era5error.npy', SI_era5error)

    msl_nceperror = np.array([surface.msl_ncep]).T * ones
    np.save(f'{outputpath}/msl_nceperror.npy', msl_nceperror)
    U10_nceperror = np.array([surface.U10_ncep]).T * ones
    np.save(f'{outputpath}/U10_nceperror.npy', U10_nceperror)
    SI_nceperror = np.array(surface.SI_ncep).reshape(-1,1) * ones
    np.save(f'{outputpath}/SI_nceperror.npy', SI_nceperror)
        
    # sea level pressure sensitivity test
    msl = np.array(surface.msl_era5)
    adjusted_msl = np.array(msl - 0.01)
    mslerrorCYCLONES_era5 = np.array([adjusted_msl]).T * ones
    np.save(f'{outputpath}/mslerrorCYCLONES_era5.npy', mslerrorCYCLONES_era5)

    msl = np.array(surface.msl_ncep)
    adjusted_msl = np.array(msl - 0.01)
    mslerrorCYCLONES_ncep = np.array([adjusted_msl]).T * ones
    np.save(f'{outputpath}/mslerrorCYCLONES_ncep.npy', mslerrorCYCLONES_ncep)

    npoints = surface.shape[0]
    # Generate random indices and add dimension for take_along_axis
    random_indices = np.random.randint(0, 4, (npoints, iters, 1))
    np.save(f'{outputpath}/random_indices.npy', random_indices)

    print("monte carlo arrays saved out")

if __name__=="__main__":
# read in data
    df = load_data()
    genmontecarlo(iters = 1000, surface=df)
