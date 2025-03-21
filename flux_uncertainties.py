"""
File: flux_uncertainties.py
----------------------------
Created on Tues Nov 19, 2024

Python script to calculate air-sea N2O fluxes,
including Monte Carlo simulation of errors due
to uncertainties in pN2Osw and pN2Oatm, and
save out.

@author: Colette Kelly (colette.kelly@whoi.edu)
"""

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import seaborn as sns
import os
from dotenv import load_dotenv
from pathlib import Path
import gsw as gsw
from gasex.airsea import L13

# functions for W14 calculation
from gasex.airsea import fsa, fsa_pC
from gasex.phys import vpress_sw
from gasex.fugacity import fugacity_factor
from gasex.sol import N2Osol_SP_pt
from gasex.sol import sol_SP_pt
import gasex.sol as sol

from initialize_paths import initialize_paths
from joblib import dump, load
import seaborn as sns
from assign_fluxes_metadata import convert_fluxesparquet, assign_fluxesmetadata
from assign_fluxes_metadata import saveoutfluxesnc, saveoutfluxespq, saveoutfluxescsv

def load_data(path_to_data):
    t = pq.read_table("datasets/n2opredictions.parquet")#(f"{path_to_data}/n2opredictions.parquet")
    df = t.to_pandas()

    return df

sns.set_context("paper", rc = {"lines.linewidth":2.5, "font.size": 12,  "font.family": "Arial"})
datapath, argopath, outputpath, era5path = initialize_paths()
data = load_data(outputpath)

data["month"] = data.JULD.dt.month # we'll use this to group data by zone and month
data["XN2Oa_sd"] = data['n2o_atm_sd']*1e-9 # this gets used in the Monte Carlo analysis

# check for NaN's to mask or drop
fluxvars = ["SP", "pt",
            "XN2Oa", "XN2Oa_sd",
            "pN2O_pred","C", "pN2Oatm",
            "U10", "msl", "SI"
           ]

vars_with_NaNs = []

for var in fluxvars:
    if len(data) != len(data.dropna(subset=[var])):
        print(f"{var} contains NaNs")
        vars_with_NaNs.append(var)

# set up dataframes containing zone areas and days in month to calculate integrated fluxes
areas = pd.DataFrame([["STZ",2.26e7],["SAZ",1.94e7],["PFZ",1.43e7],
                       ["ASZ",1.28e7],["SIZ",1.72e7],["TOTAL",8.64e7]],
            columns = ["zone","Area_km2"]).set_index("zone")

areas["m2"] = areas.Area_km2*1e6

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

# constants for converting umol/m2/day to Tg N2O-N/year
Ngpermol = 14.0067
Tgperg = 1e12
umolpermol = 1e6

counts = data[["zone", "month","msl"]].groupby(["zone", "month"]).count()
surface = data.set_index(["zone", "month"]).join(counts, rsuffix = "count").reset_index()
surface = surface.set_index("zone").join(areas["m2"]).reset_index()
surface = surface.set_index("month").join(daysinmonth["daysinmonth"]).reset_index()

# calculate fluxes from data, no error estimate, to get median Ks terms
(dP,Ks,Kb,Kc) = L13(surface.C,surface.U10,surface.SP,surface.pt,
                                        slp=surface.msl,gas='N2O',rh=1.0,
                    chi_atm=surface.XN2Oa,
                                        air_temperature=surface.pt, calculate_schmidtair=True,
                                 return_vars = ["dP","Ks","Kb","Kc"])

medKs = np.nanmedian(Ks)
medKb = np.nanmedian(Kb)
medKc = np.nanmedian(Kc)
meddP = np.nanmedian(dP)
medpN2O = np.nanmedian(surface.pN2O_pred)*1e-9

print(f"median Ks = {medKs}")
print(f"median Kb = {medKb}")
print(f"median Kc = {medKc}")
print(f"median dP = {meddP}")
print(f"median pN2O = {medpN2O}")

# compare two ways of calculating W14 flux
F1 = fsa(surface.C,surface.U10,surface.SP,surface.pt,slp=surface.msl,gas='N2O',param="W14",rh=1,
        chi_atm=surface.XN2Oa)
F1 = F1*86400*1e6

surface["ph2ov"] = vpress_sw(surface.SP,surface.pt) # atm
surface["f"]  = fugacity_factor(surface.pt,gas='N2O',slp=surface.msl)
surface["s"] = sol_SP_pt(surface.SP,surface.pt,chi_atm=surface.XN2Oa, gas='N2O',units="mM")
surface["pN2Oatm"] = surface.XN2Oa * surface.f * (surface.msl - surface.ph2ov)

pC_w = surface.pN2O_pred*1e-3 # need to convert to uatm
pC_a = surface.pN2Oatm*1e6

F2 = fsa_pC(pC_w,pC_a,surface.U10,surface.SP,surface.pt,
            gas='N2O',param="W14",chi_atm=surface.XN2Oa)
F2 = F2*86400*1e6

fig, ax = plt.subplots()
ax.plot(F1 - F2)
ax.set_title("Compare W14 flux calculation functions")
plt.tight_layout()
plt.savefig('figures/methods/compareW14.png', dpi=300)

# set up monte carlo arrays
iters = 1000
rows = len(surface)
ones = np.ones((rows,iters))

# variables containing error
pN2O_pred_reshaped = np.array(surface.pN2O_pred).reshape(-1,1)
pN2O_sd_reshaped = np.array(surface.pN2O_predstd).reshape(-1,1)
pN2Oerror = np.random.normal(loc=pN2O_pred_reshaped, scale=pN2O_sd_reshaped, size=(rows, iters))*1e-9

XN2O_reshaped = np.array(surface.XN2Oa).reshape(-1,1)
XN2O_sd_reshaped = np.array(surface.XN2Oa_sd).reshape(-1,1)
XN2Oerror = np.random.normal(loc=XN2O_reshaped, scale=XN2O_sd_reshaped, size=(rows, iters))

# variables with no error
SPerror = np.array(surface.SP).reshape(-1,1) * ones
PTerror = np.array(surface.pt).reshape(-1,1) * ones
mslerror = np.array([surface.msl]).T * ones
U10error = np.array([surface.U10]).T * ones
SIerror = np.array(surface.SI).reshape(-1,1) * ones

# sea level pressure sensitivity test
msl = np.array(surface.msl)
adjusted_msl = np.array(msl - 0.01)
mslerrorCYCLONES = np.array([adjusted_msl]).T * ones

print("monte carlo arrays set up")

################################################
### CALCULATE FLUXES FROM MONTE CARLO ARRAYS ###
################################################
# Wanninkhof 2014
# normal
FtW14output = np.ones((rows,iters))
ph2ov = vpress_sw(SPerror,PTerror) # atm
f  = fugacity_factor(PTerror,gas='N2O',slp=mslerror)
pN2Oatm = XN2Oerror * f * (mslerror - ph2ov)
pC_w = pN2Oerror*1e6 # need to convert to uatm
pC_a = pN2Oatm*1e6
FtW14 = fsa_pC(pC_w,pC_a,U10error,SPerror,PTerror,
            gas='N2O',param="W14",chi_atm=XN2Oerror)
FtW14output = (1-SIerror)*FtW14*1e6*86400

# no ice
FtW14outputnoice = FtW14*1e6*86400

# 1 atm
FtW14output1atm = np.ones((rows,iters))
ph2ov = vpress_sw(SPerror,PTerror) # atm
f  = fugacity_factor(PTerror,gas='N2O',slp=1.0)
pN2Oatm = XN2Oerror * f * (1.0 - ph2ov)
pC_w = pN2Oerror*1e6 # need to convert to uatm
pC_a = pN2Oatm*1e6
FtW14 = fsa_pC(pC_w,pC_a,U10error,SPerror,PTerror,
            gas='N2O',param="W14",chi_atm=XN2Oerror)
FtW14output1atm = (1-SIerror)*FtW14*1e6*86400

# median pN2O
FtW14outputmedN2O = np.ones((rows,iters))
ph2ov = vpress_sw(SPerror,PTerror) # atm
f  = fugacity_factor(PTerror,gas='N2O',slp=mslerror)
pN2Oatm = XN2Oerror * f * (mslerror - ph2ov)
pC_w = medpN2O*1e6 # need to convert from natm to uatm
pC_a = pN2Oatm*1e6
FtW14 = fsa_pC(pC_w,pC_a,U10error,SPerror,PTerror,
            gas='N2O',param="W14",chi_atm=XN2Oerror)
FtW14outputmedN2O = (1-SIerror)*FtW14*1e6*86400

# high winds
FtW14outputWINDS = np.ones((rows,iters))
ph2ov = vpress_sw(SPerror,PTerror) # atm
f  = fugacity_factor(PTerror,gas='N2O',slp=mslerror)
pN2Oatm = XN2Oerror * f * (mslerror - ph2ov)
pC_w = pN2Oerror*1e6 # need to convert to uatm
pC_a = pN2Oatm*1e6
FtW14 = fsa_pC(pC_w,pC_a,U10error*1.25,SPerror,PTerror,
            gas='N2O',param="W14",chi_atm=XN2Oerror)
FtW14outputWINDS = (1-SIerror)*FtW14*1e6*86400

# combined winds and no ice
FtW14outputCOMBINED = FtW14*1e6*86400

# pressure sensitivity test
FtW14outputCYCLONES = np.ones((rows,iters))
ph2ov = vpress_sw(SPerror,PTerror) # atm
f  = fugacity_factor(PTerror,gas='N2O',slp=mslerrorCYCLONES)
pN2Oatm = XN2Oerror * f * (mslerrorCYCLONES - ph2ov)
pC_w = pN2Oerror*1e6 # need to convert to uatm
pC_a = pN2Oatm*1e6
FtW14 = fsa_pC(pC_w,pC_a,U10error,SPerror,PTerror,
            gas='N2O',param="W14",chi_atm=XN2Oerror)
FtW14outputCYCLONES = (1-SIerror)*FtW14*1e6*86400

# Liang 2013
# normal
FdL13output = np.ones((rows,iters))
FcL13output = np.ones((rows,iters))
FpL13output = np.ones((rows,iters))
FtL13output = np.ones((rows,iters))


(FdL13,FcL13,FpL13) = L13(pN2Oerror,U10error,SPerror,PTerror,
                         slp=mslerror,gas='N2O',rh=1.0,chi_atm=XN2Oerror,
                         air_temperature=PTerror, calculate_schmidtair=True,
                         return_vars = ["Fd","Fc","Fp"],
                        pressure_mode = True)

FdL13output = -(1-SIerror)*FdL13*1e6*86400 # note reversed sign convention: positive = out of water
FcL13output  = -(1-SIerror)*FcL13*1e6*86400
FpL13output = -(1-SIerror)*FpL13*1e6*86400
FtL13output = FdL13output + FcL13output + FpL13output

# no ice
FtL13outputnoice = -(FdL13 + FcL13 + FpL13)*1e6*86400

# 1 atm
FtL13output1atm = np.ones((rows,iters))
(FdL13,FcL13,FpL13) = L13(pN2Oerror,U10error,SPerror,PTerror,
                         slp=1.0,gas='N2O',rh=1.0,chi_atm=XN2Oerror,
                         air_temperature=PTerror, calculate_schmidtair=True,
                         return_vars = ["Fd","Fc","Fp"],
                        pressure_mode = True)
FtL13output1atm = -(1-SIerror)*(FdL13 + FcL13 + FpL13)*1e6*86400

# median pN2O
FtL13outputmedN2O = np.ones((rows,iters))
(FdL13,FcL13,FpL13) = L13(medpN2O,U10error,SPerror,PTerror,
                         slp=mslerror,gas='N2O',rh=1.0,chi_atm=XN2Oerror,
                         air_temperature=PTerror, calculate_schmidtair=True,
                         return_vars = ["Fd","Fc","Fp"],
                        pressure_mode = True)
FtL13outputmedN2O = -(1-SIerror)*(FdL13 + FcL13 + FpL13)*1e6*86400

# median K values
FtL13outputmedK = np.ones((rows,iters))
(FdL13,FcL13,FpL13) = L13(pN2Oerror,U10error,SPerror,PTerror,
                         slp=mslerror,gas='N2O',rh=1.0,chi_atm=XN2Oerror,
                         air_temperature=PTerror, calculate_schmidtair=True,
                          Ks = medKs, Kb = medKb, Kc = medKc, dP = meddP,
                         return_vars = ["Fd","Fc","Fp"],
                        pressure_mode = True)
FtL13outputmedK = -(1-SIerror)*(FdL13 + FcL13 + FpL13)*1e6*86400

# high winds
FtL13outputWINDS = np.ones((rows,iters))
(FdL13,FcL13,FpL13) = L13(pN2Oerror,U10error*1.25,SPerror,PTerror,
                         slp=mslerror,gas='N2O',rh=1.0,chi_atm=XN2Oerror,
                         air_temperature=PTerror, calculate_schmidtair=True,
                         return_vars = ["Fd","Fc","Fp"],
                        pressure_mode = True)
FtL13outputWINDS = -(1-SIerror)*(FdL13 + FcL13 + FpL13)*1e6*86400

# combined winds and no ice
FtL13outputCOMBINED = -(FdL13 + FcL13 + FpL13)*1e6*86400

# pressure sensitivity test
FtL13outputCYCLONES = np.ones((rows,iters))
(FdL13,FcL13,FpL13) = L13(pN2Oerror,U10error,SPerror,PTerror,
                         slp=mslerrorCYCLONES,gas='N2O',rh=1.0,chi_atm=XN2Oerror,
                         air_temperature=PTerror, calculate_schmidtair=True,
                         return_vars = ["Fd","Fc","Fp"],
                        pressure_mode = True)
FtL13outputCYCLONES = -(1-SIerror)*(FdL13 + FcL13 + FpL13)*1e6*86400

print("fluxes calculated from monte carlo arrays")

# plot spread of different flux components due to Monte Carlo simulation
fig, axes = plt.subplots(1,3, figsize = (15,5))
ax = axes[0]

mu = np.mean(FdL13output[0,:])
sigma = np.std(FdL13output[0,:])
count, bins, ignored = ax.hist(FdL13output[0,:], 100, density=True, histtype="step")
ax.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
ax.set_xlabel("Fd (umol/m2/day)")

ax = axes[1]

mu = np.mean(FcL13output[0,:])
sigma = np.std(FcL13output[0,:])
count, bins, ignored = ax.hist(FcL13output[0,:], 100, density=True, histtype="step")
ax.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')

ax.set_xlabel("Fc (umol/m2/day)")

ax = axes[2]

mu = np.mean(FpL13output[0,:])
sigma = np.std(FpL13output[0,:])
count, bins, ignored = ax.hist(FpL13output[0,:], 100, density=True, histtype="step")
ax.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')

ax.set_xlabel("Fp (umol/m2/day)")
plt.tight_layout()
plt.savefig('figures/methods/FdFcFpmontecarlo.png', dpi=300)

# calculate averages across Monte Carlo arrays
surface["FtW14"] = np.mean(FtW14output, axis=1)
surface["FtW14stdev"] = 1.96*np.std(FtW14output, axis=1) # multiply by 1.96 to get 95% confidence interval

surface["FtW14noice"] = np.mean(FtW14outputnoice, axis=1)
surface["FtW14stdevnoice"] = 1.96*np.std(FtW14outputnoice, axis=1)

surface["FtW141atm"] = np.mean(FtW14output1atm, axis=1)
surface["FtW14stdev1atm"] = 1.96*np.std(FtW14output1atm, axis=1)

surface["FtW14medN2O"] = np.mean(FtW14outputmedN2O, axis=1)
surface["FtW14stdevmedN2O"] = 1.96*np.std(FtW14outputmedN2O, axis=1)

surface["FtW14WINDS"] = np.mean(FtW14outputWINDS, axis=1)
surface["FtW14stdevWINDS"] = 1.96*np.std(FtW14outputWINDS, axis=1)

surface["FtW14COMBINED"] = np.mean(FtW14outputCOMBINED, axis=1)
surface["FtW14stdevCOMBINED"] = 1.96*np.std(FtW14outputCOMBINED, axis=1)

surface["FtW14CYCLONES"] = np.mean(FtW14outputCYCLONES, axis=1)
surface["FtW14stdevCYCLONES"] = 1.96*np.std(FtW14outputCYCLONES, axis=1)

surface["FdL13"] = np.mean(FdL13output, axis=1)
surface["FcL13"] = np.mean(FcL13output, axis=1)
surface["FpL13"] = np.mean(FpL13output, axis=1)
surface["FtL13"] = np.mean(FtL13output, axis=1)

surface["FdL13stdev"] = 1.96*np.std(FdL13output, axis=1)
surface["FcL13stdev"] = 1.96*np.std(FcL13output, axis=1)
surface["FpL13stdev"] = 1.96*np.std(FpL13output, axis=1)
surface["FtL13stdev"] = 1.96*np.std(FtL13output, axis=1)

surface["FtL13noice"] = np.mean(FtL13outputnoice, axis=1)
surface["FtL13stdevnoice"] = 1.96*np.std(FtL13outputnoice, axis=1)

surface["FtL131atm"] = np.mean(FtL13output1atm, axis=1)
surface["FtL13stdev1atm"] = 1.96*np.std(FtL13output1atm, axis=1)

surface["FtL13medN2O"] = np.mean(FtL13outputmedN2O, axis=1)
surface["FtL13stdevmedN2O"] = 1.96*np.std(FtL13outputmedN2O, axis=1)

surface["FtL13medK"] = np.mean(FtL13outputmedK, axis=1)
surface["FtL13stdevmedK"] = 1.96*np.std(FtL13outputmedK, axis=1)

surface["FtL13WINDS"] = np.mean(FtL13outputWINDS, axis=1)
surface["FtL13stdevWINDS"] = 1.96*np.std(FtL13outputWINDS, axis=1)

surface["FtL13COMBINED"] = np.mean(FtL13outputCOMBINED, axis=1)
surface["FtL13stdevCOMBINED"] = 1.96*np.std(FtL13outputCOMBINED, axis=1)

surface["FtL13CYCLONES"] = np.mean(FtL13outputCYCLONES, axis=1)
surface["FtL13stdevCYCLONES"] = 1.96*np.std(FtL13outputCYCLONES, axis=1)

print("average fluxes calculated")

# take the average of W14 and L13 parameterizations
surface["Ft"] = np.mean([surface["FtW14"], surface["FtL13"]], axis=0)
diff = np.std([surface["FtW14"], surface["FtL13"]], axis=0)
surface["Ftstdev"] = np.sqrt(surface["FtW14stdev"]**2 + surface["FtL13stdev"]**2 + diff**2)

# plot standard deviations of fluxes for different parameterizations
fig, ax = plt.subplots()
ax.violinplot([surface["FtW14stdev"],surface["FtL13stdev"]])
ax.set_ylabel("flux errors ($\mu mol/m^2/d$)")
ax.set_xlabel("probability density")
ax.set_xticks([1,2])
ax.set_xticklabels(["W14", "L13"])
plt.tight_layout()
plt.savefig('figures/methods/montecarloviolin.png', dpi=300)

# plot comparison of L13 and W14 fluxes
fig, axes = plt.subplots(2,2, figsize = (10,10))
ax = axes[0,0]
ax.errorbar(surface.U10, surface["Ft"], yerr = surface["Ftstdev"],
           linestyle = "none", capsize = 5, alpha = 0.5,
           marker = "D")
ax.set_xlabel("U10 (m/s)")
ax.set_ylabel("Ft mean (umol/m2/day)")

ax = axes[0,1]
ax.errorbar(surface.U10, surface.FtL13, yerr = surface.FtL13stdev,
           linestyle = "none", capsize = 5, alpha = 0.5,
           marker = "D")
#ax.set_xlim([-0.5,5])
#ax.set_ylim([-5,2])
ax.set_xlabel("U10 (m/s)")
ax.set_ylabel("Ft L13 (umol/m2/day)")

ax = axes[1,0]
ax.errorbar(surface.U10, surface.FtW14, yerr = surface.FtW14stdev,
           linestyle = "none", capsize = 5, alpha = 0.5,
           marker = "D")
ax.set_xlabel("U10 (m/s)")
ax.set_ylabel("Ft W14 (umol/m2/day)")

ax = axes[1,1]
ax.errorbar(surface.FtL13, surface.FtW14, xerr = surface.FtL13stdev, yerr = surface.FtW14stdev,
            linestyle = "none", capsize = 5, alpha = 0.5,
            marker = "D")
ax.set_xlabel("Ft L13 (umol/m2/day)")
ax.set_ylabel("Ft W14 (umol/m2/day)")

plt.tight_layout()
plt.savefig('figures/methods/comparefluxes1.png', dpi=300)

###################################
### CALCULATE INTEGRATED FLUXES ###
###################################
surface["FtW14integrated"] = surface.FtW14*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol
surface["FtW14integratedstdev"] = surface.FtW14stdev*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol

surface["FtW14integratednoice"] = surface.FtW14noice*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol
surface["FtW14integratedstdevnoice"] = surface.FtW14stdevnoice*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol

surface["FtW14integrated1atm"] = surface.FtW141atm*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol
surface["FtW14integratedstdev1atm"] = surface.FtW14stdev1atm*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol

surface["FtW14integratedmedN2O"] = surface.FtW14medN2O*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol
surface["FtW14integratedstdevmedN2O"] = surface.FtW14stdevmedN2O*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol

surface["FtW14integratedWINDS"] = surface.FtW14WINDS*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol
surface["FtW14integratedstdevWINDS"] = surface.FtW14stdevWINDS*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol

surface["FtW14integratedCOMBINED"] = surface.FtW14COMBINED*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol
surface["FtW14integratedstdevCOMBINED"] = surface.FtW14stdevCOMBINED*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol

surface["FtW14integratedCYCLONES"] = surface.FtW14CYCLONES*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol
surface["FtW14integratedstdevCYCLONES"] = surface.FtW14stdevCYCLONES*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol

surface["FtL13integrated"] = surface.FtL13*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol
surface["FtL13integratedstdev"] = surface.FtL13stdev*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol

surface["FtL13integratednoice"] = surface.FtL13noice*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol
surface["FtL13integratedstdevnoice"] = surface.FtL13stdevnoice*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol

surface["FtL13integrated1atm"] = surface.FtL131atm*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol
surface["FtL13integratedstdev1atm"] = surface.FtL13stdev1atm*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol

surface["FtL13integratedmedN2O"] = surface.FtL13medN2O*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol
surface["FtL13integratedstdevmedN2O"] = surface.FtL13stdevmedN2O*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol

surface["FtL13integratedmedK"] = surface.FtL13medK*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol
surface["FtL13integratedstdevmedK"] = surface.FtL13stdevmedK*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol

surface["FtL13integratedWINDS"] = surface.FtL13WINDS*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol
surface["FtL13integratedstdevWINDS"] = surface.FtL13stdevWINDS*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol

surface["FtL13integratedCOMBINED"] = surface.FtL13COMBINED*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol
surface["FtL13integratedstdevCOMBINED"] = surface.FtL13stdevCOMBINED*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol

surface["FtL13integratedCYCLONES"] = surface.FtL13CYCLONES*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol
surface["FtL13integratedstdevCYCLONES"] = surface.FtL13stdevCYCLONES*surface.m2*surface.daysinmonth/surface.mslcount*(Ngpermol*2)/Tgperg/umolpermol

print("integrated fluxes calculated")

##########################################
### CALCULATE & PLOT CUMULATIVE FLUXES ###
##########################################
data_sorted = surface.sort_values("msl")
data_sorted["FtL13cumulative"] = np.cumsum(data_sorted["FtL13integrated"])
data_sorted["FtW14cumulative"] = np.cumsum(data_sorted["FtW14integrated"])
data_sorted["Ft_cumulative"] = (data_sorted["FtL13cumulative"] + data_sorted["FtW14cumulative"])/2

uncertaintiesL13 = np.array(data_sorted["FtL13integratedstdev"])
uncertaintiesL13squared = uncertaintiesL13**2
uncertaintiesL13cumulative = np.cumsum(uncertaintiesL13squared)
uncertaintiesL13 = np.sqrt(uncertaintiesL13cumulative)


uncertaintiesW14 = np.array(data_sorted["FtW14integratedstdev"])
uncertaintiesW14squared = uncertaintiesW14**2
uncertaintiesW14cumulative = np.cumsum(uncertaintiesW14squared)
uncertaintiesW14 = np.sqrt(uncertaintiesW14cumulative)

# standard deviation of cumulative sums calculated from different parameterizations
parameterizationdiff = np.std([data_sorted["FtL13cumulative"],
                              data_sorted["FtW14cumulative"]], axis=0)

Delta = np.sqrt(uncertaintiesL13**2 + uncertaintiesW14**2 + parameterizationdiff**2)

fig, ax = plt.subplots(1,1, figsize = (3.35, 3.35))
ax.fill_between(data_sorted.msl, (data_sorted["Ft_cumulative"] - Delta),
                (data_sorted["Ft_cumulative"] + Delta), color = "lightblue", zorder=0)

ax.fill_between(data_sorted.msl, (data_sorted["FtL13cumulative"] - uncertaintiesL13),
                (data_sorted["FtL13cumulative"] + uncertaintiesL13), color = "bisque", zorder=1)

ax.fill_between(data_sorted.msl, (data_sorted["FtW14cumulative"] - uncertaintiesW14),
                (data_sorted["FtW14cumulative"] + uncertaintiesW14), color = "lightcoral", zorder=1)

ax.plot(np.array(data_sorted.msl), np.array(data_sorted["Ft_cumulative"]), color = "blue",label = "real SLP")
ax.plot(np.array(data_sorted.msl), np.array(data_sorted.FtL13cumulative), label = None, color = "darkorange")
ax.plot(np.array(data_sorted.msl), np.array(data_sorted.FtW14cumulative), label = None, color = "darkred", zorder=1)

data_sorted["FtL13cumulative1atm"] = np.cumsum(data_sorted["FtL13integrated1atm"])
data_sorted["FtW14cumulative1atm"] = np.cumsum(data_sorted["FtW14integrated1atm"])
data_sorted["Ft_cumulative1atm"] = (data_sorted["FtL13cumulative1atm"] + data_sorted["FtW14cumulative1atm"])/2

uncertaintiesL13 = np.array(data_sorted["FtL13integratedstdev1atm"])
uncertaintiesL13squared = uncertaintiesL13**2
uncertaintiesL13cumulative = np.cumsum(uncertaintiesL13squared)
uncertaintiesL131atm = np.sqrt(uncertaintiesL13cumulative)

uncertaintiesW14 = np.array(data_sorted["FtW14integratedstdev1atm"])
uncertaintiesW14squared = uncertaintiesW14**2
uncertaintiesW14cumulative = np.cumsum(uncertaintiesW14squared)
uncertaintiesW141atm = np.sqrt(uncertaintiesW14cumulative)

# standard deviation of cumulative sums calculated from different parameterizations
parameterizationdiff1atm = np.std([data_sorted["FtL13cumulative1atm"],
                              data_sorted["FtW14cumulative1atm"]], axis=0)

Delta1atm = np.sqrt(uncertaintiesL131atm**2+ uncertaintiesW141atm**2 + parameterizationdiff1atm**2)

ax.fill_between(data_sorted.msl, (data_sorted["Ft_cumulative1atm"] - Delta1atm),
                (data_sorted["Ft_cumulative1atm"] + Delta1atm), color = "gray", zorder=0)

ax.fill_between(data_sorted.msl, (data_sorted["FtL13cumulative1atm"] - uncertaintiesL131atm),
                (data_sorted["FtL13cumulative1atm"] + uncertaintiesL131atm), color = "bisque", zorder=1)

ax.fill_between(data_sorted.msl, (data_sorted["FtW14cumulative1atm"] - uncertaintiesW141atm),
                (data_sorted["FtW14cumulative1atm"] + uncertaintiesW141atm), color = "lightcoral", zorder=1)

ax.plot(np.array(data_sorted.msl), np.array(data_sorted.Ft_cumulative1atm), color = "k",label = "1 atm")
ax.plot(np.array(data_sorted.msl), np.array(data_sorted.FtL13cumulative1atm), label = "L13", color = "darkorange")
ax.plot(np.array(data_sorted.msl), np.array(data_sorted.FtW14cumulative1atm), label = "W14", color = "darkred",zorder=1)

ax.legend(framealpha = 0, bbox_to_anchor=(0.0,1.0), loc = "upper left", fontsize = 12)
ax.set_ylabel(r"$N_2O$ flux (Tg N/yr)", fontsize = 12)
ax.set_xlabel("Sea level pressure (atm)", fontsize = 12)
ax.set_xlim([0.905, 1.035])
ax.set_ylim([-0.1, 1.65])
ax.set_xticks(np.linspace(0.91, 1.03, 5))
ax.set_yticks(np.linspace(0, 1.5, 7))
ax.tick_params(direction="in", top = True, right = True, labelsize=12)
plt.savefig("figures/summaryfigs/cumuflux_errors_proposal.png", dpi=300, bbox_inches = "tight")
plt.show()
plt.close()

Ftsum = data_sorted.Ft_cumulative.iloc[-1]
Ft1atmsum = data_sorted.Ft_cumulative1atm.iloc[-1]
Ftpercentage = (1-Ft1atmsum/Ftsum)*100
Ftpercentage_error = np.sqrt((Delta[-1]/Ftsum)**2+(Delta1atm[-1]/Ft1atmsum)**2)*Ftpercentage

print(f"Total flux, excluding pressure effect: {Ft1atmsum:.2}+/-{Delta1atm[-1]:.1} Tg N2O-N/yr")
print(f"Total flux, including pressure effect: {Ftsum:.2}+/-{Delta[-1]:.1} Tg N2O-N/yr")
print(f"pressure effect accounts for {round(Ftpercentage)}+/-{round(Ftpercentage_error)}% of area- and time-integrated flux")
print(f"N2O flux offsets {round(-Ftsum/2*273/3190*100)}% of CO2 flux based on 3190 Tg CO2/year (Zhong et al. 2024)")

#######################
### SAVE OUT FLUXES ###
#######################
# save out predicted pN2O and flux to directory containing intermediate data products
surface.to_parquet(f"{outputpath}/fluxes.parquet")
print("fluxes saved out")
