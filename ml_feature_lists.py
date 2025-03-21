"""
File: feature_lists.py
-------------------
Created on Weds March 12, 2025

Helper function to define different sets of
predictor features and formatted labels for plots.

@author: Colette Kelly (colette.kelly@whoi.edu)
"""

def feature_lists():
    feature_sets = {1:['AOU', 'Temp', 'Salinity', 'Nitrate'],
                    2:['DOXY', 'Temp', 'Salinity', 'Nitrate'],
                    3:['AOU', 'Temp', 'Salinity'],
                    4:['DOXY', 'Temp', 'Salinity'],
                    5:['AOU', 'Temp', 'Salinity', 'Nitrate', 'DOY_COS'],
                    6:['DOXY', 'Temp', 'Salinity', 'Nitrate','SIGMA0'],
                    192:['Salinity','Temp','DOXY','Nitrate','SIGMA0','LATITUDE','LON1','LON2','YR','DOY_COS']}

    feature_set_labels = {1:[r"$AOU$", r"$\theta$", r"$S_A$", r"$[NO_3^-]$"],
                      2:[r"$[O_2]$", r"$\theta$", r"$S_A$", r"$[NO_3^-]$"],
                      3:[r"$AOU$", r"$\theta$", r"$S_A$"],
                      4:[r"$[O_2]$", r"$\theta$", r"$S_A$"],
                      5:[r"$AOU$", r"$\theta$", r"$S_A$", r"$[NO_3^-]$",r"$cos(doy)$"],
                      6:[r"$[O_2]$", r"$\theta$", r"$S_A$", r"$[NO_3^-]$", r"$\sigma_{\theta}$"],
                      192:[r"$S_A$", r"$\theta$", r"$[O_2]$", r"$[NO_3^-]$", r"$\sigma_{\theta}$", r"$lat$", r"$lon1$", r"$lon2$", r"$yr$", r"$cos(doy)$"]}
    
    argo_feature_sets = {1:['AOU', 'Temp', 'Salinity', 'Nitrate'],
                    2:['DOXY_ADJUSTED', 'Temp', 'Salinity', 'Nitrate'],
                    3:['AOU', 'Temp', 'Salinity'],
                    4:['DOXY_ADJUSTED', 'Temp', 'Salinity'],
                    5:['AOU', 'Temp', 'Salinity', 'Nitrate', 'DOY_COS'],
                    6:['DOXY_ADJUSTED', 'Temp', 'Salinity', 'Nitrate', 'sigma0'],
                    192:['Salinity','Temp','DOXY_ADJUSTED','Nitrate','sigma0','LATITUDE','LON1','LON2','YR','DOY_COS']}

    return feature_sets, feature_set_labels, argo_feature_sets

if __name__=="__main__":
    feature_sets, feature_set_labels = feature_lists()
    print(feature_sets, feature_set_labels)
