#!/usr/bin/env python3
"""
Calculate air-sea flux for a given parameterization of air-sea
gas exchange and given sensitivity test.

Created Mon Jun 23 2025
@author: Colette Kelly
colette.kelly@whoi.edu
"""
import numpy as np
from gasex.phys import vpress_sw
from gasex.fugacity import fugacity_factor
from gasex.airsea import fsa, fsa_pC, kgas
from gasex.airsea import L13
from joblib import Parallel, delayed # parallel processing
import pandas as pd
import pyarrow.parquet as pq
from initialize_paths import initialize_paths
from gasex.diff import schmidt

def loadmontecarlo(outputpath):
    pN2Oerror = np.load(f'{outputpath}/pN2Oerror.npy')
    pN2Oerror_bias19 = np.load(f'{outputpath}/pN2Oerror_bias19.npy')
    pN2Oerror_bias27 = np.load(f'{outputpath}/pN2Oerror_bias27.npy')

    XN2Oerror = np.load(f'{outputpath}/XN2Oerror.npy')
    SPerror = np.load(f'{outputpath}/SPerror.npy')
    PTerror = np.load(f'{outputpath}/PTerror.npy')
    
    msl_era5error = np.load(f'{outputpath}/msl_era5error.npy')
    U10_era5error = np.load(f'{outputpath}/U10_era5error.npy')
    SI_era5error = np.load(f'{outputpath}/SI_era5error.npy')
    mslerrorCYCLONES_era5 = np.load(f'{outputpath}/mslerrorCYCLONES_era5.npy')

    msl_nceperror = np.load(f'{outputpath}/msl_nceperror.npy')
    U10_nceperror = np.load(f'{outputpath}/U10_nceperror.npy')
    SI_nceperror = np.load(f'{outputpath}/SI_nceperror.npy')
    mslerrorCYCLONES_ncep = np.load(f'{outputpath}/mslerrorCYCLONES_ncep.npy')

    random_indices = np.load(f'{outputpath}/random_indices.npy')

    return (pN2Oerror,pN2Oerror_bias19, pN2Oerror_bias27,
            XN2Oerror, SPerror, PTerror,
        msl_era5error, U10_era5error, SI_era5error, mslerrorCYCLONES_era5,
        msl_nceperror, U10_nceperror, SI_nceperror, mslerrorCYCLONES_ncep,
           random_indices)

def loadbiasedmontecarlo(outputpath):
    pN2Oarrays = []
    for i in range(-10,12,2):
        pN2Oerror_biased = np.load(f'{outputpath}/pN2Oerror_trainbias{i}.npy')
        pN2Oarrays.append(pN2Oerror_biased)
    return pN2Oarrays

def calckvals(outputpath, median=False):
    t = pq.read_table(f"{outputpath}/n2opredictions.parquet")
    surface = t.to_pandas()
    # calculate fluxes from data, no error estimate, to get median Ks terms
    (dP_era5,Ks_era5,Kb_era5,Kc_era5) = L13(surface.C_era5,surface.U10_era5,surface.SP,surface.pt,
        slp=surface.msl_era5, gas='N2O', rh=1.0, chi_atm=surface.XN2Oa,
        air_temperature=surface.pt, calculate_schmidtair=True,
        return_vars = ["dP","Ks","Kb","Kc"])
    Sc = schmidt(surface.SP,surface.pt,gas="n2o")
    Kw14_era5 = kgas(surface.U10_era5,Sc,param="W14")
    
    # calculate fluxes from data, no error estimate, to get median Ks terms
    (dP_ncep,Ks_ncep,Kb_ncep,Kc_ncep) = L13(surface.C_ncep,surface.U10_ncep,surface.SP,surface.pt,
        slp=surface.msl_ncep, gas='N2O', rh=1.0, chi_atm=surface.XN2Oa,
        air_temperature=surface.pt, calculate_schmidtair=True,
        return_vars = ["dP","Ks","Kb","Kc"])
    Sc = schmidt(surface.SP,surface.pt,gas="n2o")
    Kw14_ncep = kgas(surface.U10_ncep,Sc,param="W14")

    if median == False:
        # means
        return (np.nanmean(Ks_era5),np.nanmean(Kb_era5),np.nanmean(Kc_era5),np.nanmean(dP_era5),np.nanmean(Kw14_era5),
               np.nanmean(Ks_ncep),np.nanmean(Kb_ncep),np.nanmean(Kc_ncep),np.nanmean(dP_ncep),np.nanmean(Kw14_ncep))
    else:
        return (np.nanmedian(Ks_era5),np.nanmedian(Kb_era5),np.nanmedian(Kc_era5),np.nanmedian(dP_era5),np.nanmedian(Kw14_era5),
               np.nanmedian(Ks_ncep),np.nanmedian(Kb_ncep),np.nanmedian(Kc_ncep),np.nanmedian(dP_ncep),np.nanmedian(Kw14_ncep))

def calcmedmsl(outputpath, median=False):
    t = pq.read_table(f"{outputpath}/n2opredictions.parquet")
    surface = t.to_pandas()
    
    if median == False:
        return (np.nanmean(surface.msl_era5), np.nanmean(surface.msl_ncep))
    else:
        return (np.nanmedian(surface.msl_era5), np.nanmedian(surface.msl_ncep))

def calcmedpN2O(outputpath, median=False):
    t = pq.read_table(f"{outputpath}/n2opredictions.parquet")
    surface = t.to_pandas()
    if median == False:
        return np.nanmean(surface.pN2O_pred)*1e-9
    else:
        return np.nanmedian(surface.pN2O_pred)*1e-9

def calcflux(rows, iters, test, outputpath, SPerror, PTerror, XN2Oerror, pN2Oerror,
             U10_era5error, msl_era5error,
             U10_nceperror, msl_nceperror,
             SI_era5error,
             random_indices,
             Ks_era5=None, Kb_era5=None,Kc_era5=None, dP_era5=None, Kw14_era5=None,
            Ks_ncep=None, Kb_ncep=None,Kc_ncep=None, dP_ncep=None, Kw14_ncep=None):

    cases = 4
    n_methods = 4
    Foutput = np.ones((rows, iters, cases))
    
    # case 1: W14, ERA5
    ph2ov = vpress_sw(SPerror,PTerror) # atm
    f  = fugacity_factor(PTerror,gas='N2O',slp=msl_era5error)
    pN2Oatm = XN2Oerror * f * (msl_era5error - ph2ov)
    pC_w = pN2Oerror*1e6 # need to convert to uatm
    pC_a = pN2Oatm*1e6
    FtW14 = fsa_pC(pC_w,pC_a,U10_era5error,SPerror,PTerror,
                gas='N2O',param="W14",chi_atm=XN2Oerror, k = Kw14_era5)
    Foutput[:,:,0] = (1-SI_era5error)*FtW14*1e6*86400
    
    # case 2: W14, ncep
    ph2ov = vpress_sw(SPerror,PTerror) # atm
    f  = fugacity_factor(PTerror,gas='N2O',slp=msl_nceperror)
    pN2Oatm = XN2Oerror * f * (msl_nceperror - ph2ov)
    pC_w = pN2Oerror*1e6 # need to convert to uatm
    pC_a = pN2Oatm*1e6
    FtW14 = fsa_pC(pC_w,pC_a,U10_nceperror,SPerror,PTerror,
                gas='N2O',param="W14",chi_atm=XN2Oerror, k = Kw14_ncep)
    Foutput[:,:,1] = (1-SI_era5error)*FtW14*1e6*86400 # still use ERA5 sea ice because NCEP is just 0 or 1
    
    # case 3: L13, ERA5
    (FdL13,FcL13,FpL13) = L13(pN2Oerror,U10_era5error,SPerror,PTerror,
                             slp=msl_era5error, gas='N2O',rh=1.0,chi_atm=XN2Oerror,
                             air_temperature=PTerror, calculate_schmidtair=True,
                             Ks = Ks_era5, Kb = Kb_era5, Kc = Kc_era5, dP = dP_era5,
                             return_vars = ["Fd","Fc","Fp"],
                            pressure_mode = True)
    Foutput[:,:,2] = -(1-SI_era5error)*(FdL13 + FcL13 + FpL13)*1e6*86400
    
    # case 4: L13, NCEP
    (FdL13,FcL13,FpL13) = L13(pN2Oerror,U10_nceperror,SPerror,PTerror,
                             slp=msl_nceperror, gas='N2O',rh=1.0,chi_atm=XN2Oerror,
                             air_temperature=PTerror, calculate_schmidtair=True,
                             Ks = Ks_ncep, Kb = Kb_ncep, Kc = Kc_ncep, dP = dP_ncep,
                             return_vars = ["Fd","Fc","Fp"],
                            pressure_mode = True)
    Foutput[:,:,3] = -(1-SI_era5error)*(FdL13 + FcL13 + FpL13)*1e6*86400

    # save out means and stdevs for individual methods
    means = np.mean(Foutput, axis = 1)
    stdevs = np.std(Foutput, axis = 1)

    # case 5: OPTION 1: randomly switch between products and parameterizations
    Frand = np.take_along_axis(Foutput, random_indices, axis=2).squeeze(axis=2) # shape = (rows, iters)
    Frandmean = np.mean(Frand, axis = 1)
    Frandstdev = np.std(Frand, axis = 1)

    # case 5: OPTION 2: take the mean of all methods
    # average across Monte Carlo iterations
    method_means = np.mean(Foutput, axis=1)  # Shape: (rows, methods)
    # average across methods
    combined_mean = np.mean(method_means, axis=1)  # Shape: (rows,)

    # will store a covariance matrix for each profile here
    method_covariances = np.zeros((rows, n_methods, n_methods))

    # loop through profiles one at a time
    for i in range(rows):
        # for each profile i, get all MC samples for this profile across methods
        profile_samples = Foutput[i, :, :]  # Shape: (iters, methods)
        
        # calculate the covariance matrix between the columns (methods)
        # the different methods will all predict a higher flux for a Monte Carlo iteration with higher pN2Osw,
        # so the covariance matrix will all be positive
        if profile_samples.shape[0] > 1:
            method_covariances[i, :, :] = np.cov(profile_samples.T)
    
    # take the average of all the covariance matrices for all the profiles
    avg_method_covariance = np.mean(method_covariances, axis=0)
    
    # diagonal elements: individual method uncertainties
    # off-diagonal elements: method covariances
    method_stds = np.sqrt(np.diag(avg_method_covariance))
    # correlation rⱼₖ = σⱼₖ / (σⱼ × σₖ)
    method_correlation_matrix = avg_method_covariance / np.outer(method_stds, method_stds)
    
    print("Method correlation matrix:")
    method_names = ['W14_ERA5', 'W14_NCEP', 'L13_ERA5', 'L13_NCEP']
    for i, name_i in enumerate(method_names):
        print(f"{name_i:10s}: ", end="")
        for j, name_j in enumerate(method_names):
            print(f"{method_correlation_matrix[i,j]:6.3f}", end=" ")
        print()
    
    # Calculate the combined uncertainty for each profile, including covariance between methods
    combined_uncertainty = np.zeros(rows)
    for i in range(rows):
        profile_cov_matrix = method_covariances[i, :, :]
        # σ²y = σ²a(∂y/∂a)² + σ²b(∂y/∂b)² + 2σ²ab(∂y/∂a)(∂y/∂b) ...
        # since F̄ = (1/n_methods) × (F₁ + F₂ + ... + Fₙ), then (∂F̄/∂F₁) = (1/n_methods)
        # all elements of covariance matrix are multiplied by (1/n_methods)^2
        # so the combined variance = sum of covariance matrix * (1/n_methods)^2
        combined_variance = np.sum(profile_cov_matrix) / (n_methods ** 2)
        # ...and the uncertainty is just the square root
        combined_uncertainty[i] = np.sqrt(max(0, combined_variance))
    
    # save out
    # case 1: W14, ERA5
    np.save(f'{outputpath}/fluxtests/{test}era5W14_mean.npy', means[:,0])
    np.save(f'{outputpath}/fluxtests/{test}era5W14_stdev.npy', stdevs[:,0])
   
    # case 2: W14, ncep
    np.save(f'{outputpath}/fluxtests/{test}ncepW14_mean.npy', means[:,1])
    np.save(f'{outputpath}/fluxtests/{test}ncepW14_stdev.npy', stdevs[:,1])
    
    # case 3: L13, ERA5
    np.save(f'{outputpath}/fluxtests/{test}era5L13_mean.npy', means[:,2])
    np.save(f'{outputpath}/fluxtests/{test}era5L13_stdev.npy', stdevs[:,2])

    # case 4: L13, NCEP
    np.save(f'{outputpath}/fluxtests/{test}ncepL13_mean.npy', means[:,3])
    np.save(f'{outputpath}/fluxtests/{test}ncepL13_stdev.npy', stdevs[:,3])

    # case 5: randomly select between L13 and W14; ERA5 and NCEP
    np.save(f'{outputpath}/fluxtests/{test}random_mean.npy', Frandmean)
    np.save(f'{outputpath}/fluxtests/{test}random_stdev.npy', Frandstdev)

    # OR: combined Monte Carlo uncertainties from all four parameterization/wind speed combos + covariances
    np.save(f'{outputpath}/fluxtests/{test}combined_mean.npy', combined_mean)
    np.save(f'{outputpath}/fluxtests/{test}combined_stdev.npy', combined_uncertainty)
    
    return Foutput, Frandmean, Frandstdev, combined_mean, combined_uncertainty, method_correlation_matrix, method_covariances

def sensitivitytest(montecarloarrays, outputpath, test="observed"):

    (pN2Oerror,pN2Oerror_bias19, pN2Oerror_bias27, XN2Oerror, SPerror, PTerror,
        msl_era5error, U10_era5error, SI_era5error, mslerrorCYCLONES_era5,
        msl_nceperror, U10_nceperror, SI_nceperror, mslerrorCYCLONES_ncep,
    random_indices) = montecarloarrays
    s = pN2Oerror.shape
    rows = s[0]
    iters = s[1]

    if test == "observed":
        Foutput, Frandmean, Frandstdev, combined_mean, combined_uncertainty, method_correlation_matrix, method_covariances  = calcflux(rows, iters, test,
            outputpath, SPerror, PTerror, XN2Oerror, pN2Oerror,
                                                        U10_era5error, msl_era5error,
                                                        U10_nceperror, msl_nceperror,
                                                        SI_era5error,
                                                        random_indices)

    elif test == "baseline":
        (medKs_era5,medKb_era5,medKc_era5,meddP_era5,medKw14_era5,
        medKs_ncep,medKb_ncep,medKc_ncep,meddP_ncep,medKw14_ncep) = calckvals(outputpath, median=True)
        
        Foutput, Frandmean, Frandstdev, combined_mean, combined_uncertainty, method_correlation_matrix, method_covariances  = calcflux(rows, iters, test,
            outputpath, SPerror, PTerror, XN2Oerror, pN2Oerror,
                                                        U10_era5error, 1.0,
                                                        U10_nceperror, 1.0,
                                                        SI_era5error,
                                                        random_indices,
                                                        Ks_era5=medKs_era5, Kb_era5=medKb_era5,Kc_era5=medKc_era5, dP_era5=meddP_era5, Kw14_era5=medKw14_era5,
                                                        Ks_ncep=medKs_ncep, Kb_ncep=medKb_ncep,Kc_ncep=medKc_ncep, dP_ncep=meddP_ncep, Kw14_ncep=medKw14_ncep)

    elif test == "noice":
        Foutput, Frandmean, Frandstdev, combined_mean, combined_uncertainty, method_correlation_matrix, method_covariances  = calcflux(rows, iters, test,
            outputpath, SPerror, PTerror, XN2Oerror, pN2Oerror,
                                                        U10_era5error, msl_era5error,
                                                        U10_nceperror, msl_nceperror,
                                                        0.0,
                                                        random_indices)
    
    elif test == "1atm":
        Foutput, Frandmean, Frandstdev, combined_mean, combined_uncertainty, method_correlation_matrix, method_covariances  = calcflux(rows, iters, test,
            outputpath, SPerror, PTerror, XN2Oerror, pN2Oerror,
                                                        U10_era5error, 1.0,
                                                        U10_nceperror, 1.0,
                                                        SI_era5error,
                                                        random_indices)
    elif test == "medmsl":
        (medmsl_era5, medmsl_ncep) = calcmedmsl(outputpath, median=True)
        Foutput, Frandmean, Frandstdev, combined_mean, combined_uncertainty, method_correlation_matrix, method_covariances  = calcflux(rows, iters, test,
            outputpath, SPerror, PTerror, XN2Oerror, pN2Oerror,
                                                        U10_era5error, medmsl_era5,
                                                        U10_nceperror, medmsl_ncep,
                                                        SI_era5error,
                                                        random_indices)
    elif test == "meanmsl":
        (medmsl_era5, medmsl_ncep) = calcmedmsl(outputpath, median=False)
        Foutput, Frandmean, Frandstdev, combined_mean, combined_uncertainty, method_correlation_matrix, method_covariances  = calcflux(rows, iters, test,
            outputpath, SPerror, PTerror, XN2Oerror, pN2Oerror,
                                                        U10_era5error, medmsl_era5,
                                                        U10_nceperror, medmsl_ncep,
                                                        SI_era5error,
                                                        random_indices)

    elif test == "medK":
        (medKs_era5,medKb_era5,medKc_era5,meddP_era5,medKw14_era5,
        medKs_ncep,medKb_ncep,medKc_ncep,meddP_ncep,medKw14_ncep) = calckvals(outputpath, median=True)
       
        Foutput, Frandmean, Frandstdev, combined_mean, combined_uncertainty, method_correlation_matrix, method_covariances  = calcflux(rows, iters, test,
            outputpath, SPerror, PTerror, XN2Oerror, pN2Oerror,
                                                        U10_era5error, msl_era5error,
                                                        U10_nceperror, msl_nceperror,
                                                        SI_era5error,
                                                        random_indices,
                                                        Ks_era5=medKs_era5, Kb_era5=medKb_era5,Kc_era5=medKc_era5, dP_era5=meddP_era5, Kw14_era5=medKw14_era5,
                                                        Ks_ncep=medKs_ncep, Kb_ncep=medKb_ncep,Kc_ncep=medKc_ncep, dP_ncep=meddP_ncep, Kw14_ncep=medKw14_ncep)

    elif test == "meanK":
        (medKs_era5,medKb_era5,medKc_era5,meddP_era5,medKw14_era5,
        medKs_ncep,medKb_ncep,medKc_ncep,meddP_ncep,medKw14_ncep) = calckvals(outputpath, median=False)
       
        Foutput, Frandmean, Frandstdev, combined_mean, combined_uncertainty, method_correlation_matrix, method_covariances  = calcflux(rows, iters, test,
            outputpath, SPerror, PTerror, XN2Oerror, pN2Oerror,
                                                        U10_era5error, msl_era5error,
                                                        U10_nceperror, msl_nceperror,
                                                        SI_era5error,
                                                        random_indices,
                                                        Ks_era5=medKs_era5, Kb_era5=medKb_era5,Kc_era5=medKc_era5, dP_era5=meddP_era5, Kw14_era5=medKw14_era5,
                                                        Ks_ncep=medKs_ncep, Kb_ncep=medKb_ncep,Kc_ncep=medKc_ncep, dP_ncep=meddP_ncep, Kw14_ncep=medKw14_ncep)

    elif test == "medN2O":
        medpN2O = calcmedpN2O(outputpath, median=True)
        Foutput, Frandmean, Frandstdev, combined_mean, combined_uncertainty, method_correlation_matrix, method_covariances  = calcflux(rows, iters, test,
            outputpath, SPerror, PTerror, XN2Oerror, medpN2O,
                                                        U10_era5error, msl_era5error,
                                                        U10_nceperror, msl_nceperror,
                                                        SI_era5error,
                                                        random_indices)

    elif test == "meanN2O":
        meanpN2O = calcmedpN2O(outputpath, median=False)
        Foutput, Frandmean, Frandstdev, combined_mean, combined_uncertainty, method_correlation_matrix, method_covariances  = calcflux(rows, iters, test,
            outputpath, SPerror, PTerror, XN2Oerror, meanpN2O,
                                                        U10_era5error, msl_era5error,
                                                        U10_nceperror, msl_nceperror,
                                                        SI_era5error,
                                                        random_indices)

    elif test == "WINDS":
        # high winds
        Foutput, Frandmean, Frandstdev, combined_mean, combined_uncertainty, method_correlation_matrix, method_covariances  = calcflux(rows, iters, test,
            outputpath, SPerror, PTerror, XN2Oerror, pN2Oerror,
                                                        U10_era5error*1.25, msl_era5error,
                                                        U10_nceperror*1.25, msl_nceperror,
                                                        SI_era5error,
                                                        random_indices)

    elif test == "COMBINED":
        # high winds and no ice
        Foutput, Frandmean, Frandstdev, combined_mean, combined_uncertainty, method_correlation_matrix, method_covariances  = calcflux(rows, iters, test,
            outputpath, SPerror, PTerror, XN2Oerror, pN2Oerror,
                                                        U10_era5error*1.25, msl_era5error,
                                                        U10_nceperror*1.25, msl_nceperror,
                                                        0.0,
                                                        random_indices)
    elif test == "CYCLONES":
        # lower barometric pressure by 0.01
        Foutput, Frandmean, Frandstdev, combined_mean, combined_uncertainty, method_correlation_matrix, method_covariances  = calcflux(rows, iters, test,
            outputpath, SPerror, PTerror, XN2Oerror, pN2Oerror,
                                                        U10_era5error, mslerrorCYCLONES_era5,
                                                        U10_nceperror, mslerrorCYCLONES_ncep,
                                                        SI_era5error,
                                                        random_indices)

    print(f"{test} saved out")

def generateinputs():
    parameterizations = ["W14", "L13"]
    windproducts = ["era5", "ncep"]
    tests = ["observed", "baseline", "noice", "1atm", "medmsl", "meanmsl", "medK", "meanK", "medN2O",
    #"meanN2O", # for whatever reason, this one throws an error
    "WINDS", "COMBINED", "CYCLONES"
    ]
    inputs = [[p,w, t] for p in parameterizations for w in windproducts for t in tests]
    print(f"inputs: {inputs}")
    return inputs

def main():
    datapath, argopath, outputpath, era5path = initialize_paths()
    montecarloarrays = loadmontecarlo(outputpath)
    biasedpN2Oarrays = loadbiasedmontecarlo(outputpath)
    (pN2Oerror,pN2Oerror_bias19, pN2Oerror_bias27, XN2Oerror, SPerror, PTerror,
        msl_era5error, U10_era5error, SI_era5error, mslerrorCYCLONES_era5,
        msl_nceperror, U10_nceperror, SI_nceperror, mslerrorCYCLONES_ncep,
    random_indices) = montecarloarrays
    s = pN2Oerror.shape
    rows = s[0]
    iters = s[1]
    print(iters)
    
    Foutput, Frandmean, Frandstdev, combined_mean, combined_uncertainty, method_correlation_matrix, method_covariances = calcflux(rows, iters, "observed",
        outputpath,
             SPerror, PTerror, XN2Oerror, pN2Oerror,
             U10_era5error, msl_era5error,
             U10_nceperror, msl_nceperror,
             SI_era5error,
             random_indices,
             Ks_era5=None, Kb_era5=None,Kc_era5=None, dP_era5=None, Kw14_era5=None,
            Ks_ncep=None, Kb_ncep=None,Kc_ncep=None, dP_ncep=None, Kw14_ncep=None)

    # first, calculate fluxes for bias tests

    tests = [f"_trainbias{i}" for i in range(-10,12,2)]
    '''
    ntests = len(tests)
    Parallel(n_jobs = ntests)(delayed(calcflux)(rows, iters, test,
                     outputpath,
                     SPerror, PTerror, XN2Oerror, biasedpN2Oarrays[count],
                     U10_era5error, msl_era5error,
                     U10_nceperror, msl_nceperror,
                     SI_era5error,
                     random_indices,
                     Ks_era5=None, Kb_era5=None,Kc_era5=None, dP_era5=None, Kw14_era5=None,
                    Ks_ncep=None, Kb_ncep=None,Kc_ncep=None, dP_ncep=None, Kw14_ncep=None) for count, test in enumerate(tests))
    
    '''
    for count, test in enumerate(tests):
        print(f"running test {test}")
        Foutput, Frandmean, Frandstdev, combined_mean, combined_uncertainty, method_correlation_matrix, method_covariances = calcflux(rows, iters, test,
                     outputpath,
                     SPerror, PTerror, XN2Oerror, biasedpN2Oarrays[count],
                     U10_era5error, msl_era5error,
                     U10_nceperror, msl_nceperror,
                     SI_era5error,
                     random_indices,
                     Ks_era5=None, Kb_era5=None,Kc_era5=None, dP_era5=None, Kw14_era5=None,
                    Ks_ncep=None, Kb_ncep=None,Kc_ncep=None, dP_ncep=None, Kw14_ncep=None)
    '''

    tests = ["observed", "baseline", "noice", "1atm", "medmsl", "meanmsl", "medK", "meanK", "medN2O",
        "meanN2O", "WINDS", "COMBINED", "CYCLONES"]
    #ntests = len(tests)
    #Parallel(n_jobs = ntests)(delayed(sensitivitytest)(montecarloarrays, outputpath, test=t) for t in tests)
    
    for t in tests:
        print(f"running test {t}")
        sensitivitytest(montecarloarrays, outputpath, test=t)


    inputs = generateinputs()
    ntests = len(inputs)
    Parallel(n_jobs = ntests)(delayed(sensitivitytest)(montecarloarrays, outputpath,
        parameterization = inputs[i][0],
        windproduct=inputs[i][1],
        test = inputs[i][2]) for i in range(ntests))
    
    #sensitivitytest(pN2Oerror, XN2Oerror, SPerror, PTerror, mslerror, U10_era5error, U10_nceperror, SIerror, mslerrorCYCLONES,
    #test="observed", windproduct="era5", parameterization = "W14")
    '''
if __name__=="__main__":
    main()
