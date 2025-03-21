import gsw
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def GetGOSHIPN20data(datapath, target_col):
    go = pd.read_parquet("datasets/goshipdataset.parquet")#(f"{datapath}allgoshipn2o.parquet")

    ## QC
    # during data compilation, already filtered out data where BTLNBR, CTDSAL, CTDTMP, and CTDPRS were flagged as bad
    # here we further filter out data where N2O and nitrate were flagged as bad
    badflags = [
        1.0, # sample was drawn from water bottle but analysis not received
        3.0, # questionable measurement
        4.0, # bad measurement
        5.0, # not reported
        9.0 # sample not drawn for this measurement from this bottle
    ]

    bad_mask = np.isin([
        go['N2O_FLAG_W'],
        go['NITRAT_FLAG_W']
                   ],
                   badflags).max(axis=0) # "max" here means False + True = True

    go = go[~bad_mask]

    ## Format date
    tlist = []
    for d in go.DATE:
        tlist.append(datetime.strptime(str(d), '%Y%m%d.0'))
    go.DATE = tlist

    # Add additional random parameters
    go["refdays"] = [pd.Timedelta(di,'D')/pd.Timedelta(1,'D') for di in go["DATE"]-datetime(2000,1,1)]

    go["yr"] = [pd.Timestamp(fi).year for fi in go.loc[:,'DATE'].values]
    
    doy = np.array([pd.Timestamp(fi).dayofyear for fi in go.loc[:,'DATE'].values])
    go["doy"] = doy
    go["doy_cos"] = np.cos(2*np.pi*doy/365.25)
    go["doy_sin"] = np.sin(2*np.pi*doy/365.25)
    go["doy_rad"] = 2*np.pi*doy/365.25
    
    go["lon"] = np.cos(np.pi*(go["LONGITUDE"])/180)
    go["lon1"] = np.cos(np.pi*(go["LONGITUDE"]-110)/180)
    go["lon2"] = np.cos(np.pi*(go["LONGITUDE"]-20)/180)

    #z is depth, using gsw/TEOS toolbox
    go["z"] = -gsw.z_from_p(go.CTDPRS.to_numpy(),go.LATITUDE.to_numpy())

    go['SPICE'] = gsw.spiciness0(go.loc[:,'SA'], go.loc[:,'CT'])

    # note that conservative temp and absolute salinity have been renamed to "Temp" and "Salinity"
    # Note: Ellen renamed these because she easily gets confused lol
    feature_cols = ['EXPOCODE','STNNBR','SECT_ID',
                    'LATITUDE','LONGITUDE','lon','lon1','lon2',
                    'DATE','yr','doy',
                    'refdays','doy_cos','doy_sin','doy_rad',
                    'CTDSAL','CTDTMP','z','CTDOXY',
                    'SA','PT','CT','sigma0',
                    'AOU','NITRAT','SPICE', 
                    #'BIOME_NUM','BIOME_ABREV' # don't include biomes for now - some biome labels are NaN
                   ]
    
    new_cols = ['EXPOCODE','STNNBR','SECT_ID',
                'LATITUDE','LONGITUDE','LON','LON1','LON2',
                'DATE','YR','DOY',
                'DAYS2000','DOY_COS','DOY_SIN','DOY_RAD',
                'PSAL','TEMP','Z','DOXY',
                'SA','PT','CT','SIGMA0',
                'AOU','NITRATE','SPICE',
                #'BIOME_NUM','BIOME_ABREV' # don't include biomes for now - some biome labels are NaN
               ]
    
    # print(len(feature_cols), len(new_cols))
    go=go.rename(columns=dict(zip(feature_cols,new_cols)))
    
    # Reformat the data so it is useful
    all_cols = new_cols + [target_col]
    go = go[all_cols].dropna(axis='rows') # bad flagged data are set to NaN so this removes them
    go=go[(go["Z"] > 0)]  #ALL DEPTHS
    go = go[go[target_col] > 0]
    
    return go

def TrainTestSplitByStations(go, test_split, random_state=100):
    
    # Randomly withhold specified proportion of stations
    # Need to keep 'EXPOCODE' and 'STNNBR' to uniquely identify each station.
    all_stns = go.loc[:,['EXPOCODE','STNNBR','pN2O']].groupby(['EXPOCODE','STNNBR']).count().index.values
    # Split the stations into training and validation sets with the split specified by kwarg "test_split"
    training_stations, validation_stations = train_test_split(all_stns, test_size = test_split, random_state = random_state)
    
    # Extract 'EXPOCODE' and 'STNNBR' for both training and validation sets.
    training_expocode = np.array([vi[0] for vi in training_stations])
    training_stnnbr = np.array([vi[1] for vi in training_stations])
    validation_expocode = np.array([vi[0] for vi in validation_stations])
    validation_stnnbr = np.array([vi[1] for vi in validation_stations])

    training_inds = []
    test_inds = []

    # Iterate over all rows in 'go' to assign indices to either training or validation sets.
    for i in np.arange(go.shape[0]):
        expocode = go.loc[:,'EXPOCODE'].values[i]
        stnnbr = go.loc[:,'STNNBR'].values[i]

        # Check if the current station belongs to the training set.
        if np.where((training_expocode==expocode) & (training_stnnbr==stnnbr))[0].shape[0]>0:
            # check that they are on the same line 
            training_inds = training_inds+ [i]
        # Check if the current station belongs to the validation set.
        elif np.where((validation_expocode==expocode) & (validation_stnnbr==stnnbr))[0].shape[0]>0:
            test_inds = test_inds+ [i]
            
    # Create separate DataFrames for training and validation sets.
    go_test = go.iloc[test_inds, :]
    go_training = go.iloc[training_inds, :]
    
    return go_training, training_inds, go_test, test_inds

def TrainRandomForest(X, Y,n_cores, param_dict, random_state = 100, verbose = True):

    # Set up work flow to test different parameter pairings
    # if tune_hyperparameters == True, will use GridSearchCV to find
    # optimal hyperparameters trained on the testing data

    # n_cores: # of computational cores available None=1, or as many as specifiecd

    RF = RandomForestRegressor(n_estimators=param_dict['n_estimators'],
                               max_features=param_dict['max_features'],
                               max_depth=param_dict['max_depth'],
                               min_samples_split=param_dict['min_samples_split'],
                               min_samples_leaf=param_dict['min_samples_leaf'],
                                random_state = random_state, n_jobs=n_cores,
                              criterion = param_dict['criterion'])

    #'fit' implements RF hyperparameters on training dataset
    # train random forest
    #RF = RF.fit(X_train,Y_train)
    RF = RF.fit(X, Y)

    if verbose: print (f'Accuracy - : {RF.score(X,Y):.3f}')

    return RF