"""
File: trainrf_v2.py
-------------------
Created on Weds March 12, 2025

Train four core Random Forest models on full
training dataset (e.g., training + validation
data) to predict N2O from T, S, O2, and NO3-.

@author: Colette Kelly (colette.kelly@whoi.edu)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ml fxns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import metrics
from joblib import dump

# helper functions
from initialize_paths import initialize_paths
from ml_feature_lists import feature_lists
from plottraintest import plottraintest
from ellen_ml_fxns import GetGOSHIPN20data, TrainTestSplitByStations

def load_data(datapath, target_col):
    """
    DESCRIPTION
    -------------
    Load and preprocess the GO-SHIP N2O dataset.

    Filters out flagged data and splits it into training and test sets.

    INPUTS:
    ----------
    datapath    Path to the dataset file
    target_col  Column name of the target variable

    OUTPUT:
    ----------
    go_training    Training dataset
    go_test        Test dataset
    """
    print(f"loading data from {datapath}")
    go_v2 = GetGOSHIPN20data(datapath, target_col)
    go_v2 = go_v2.rename(columns = {'CT':'Temp','SA':'Salinity','NITRATE':'Nitrate'})

    # filter out flier
    flaggedsample = go_v2[(go_v2.EXPOCODE == "33RR20230722")&(go_v2.STNNBR == 40)&(go_v2.Z <=30)].index[0]
    go_v2 = go_v2.reset_index().rename(columns = {"index":"sampleID"})
    go_v2 = go_v2[go_v2["sampleID"] != flaggedsample]

    # Split data into training and testing sets based on station
    go_training, _, go_test, _ = TrainTestSplitByStations(go_v2, 0.2, random_state=100)

    return go_training, go_test

def gen_RF(xtrain, xtest, ytrain, ytest, n_cores=16,
           n_estimators=600, min_samples_leaf=10, min_samples_split = 2, random_state=100):
    """
    DESCRIPTION
    -------------
    Train a Random Forest model for predicting pN2O.

    Fits the model on the training dataset and evaluates its performance on the test dataset.

    INPUTS:
    ----------
    xtrain            Training feature matrix
    xtest             Test feature matrix
    ytrain            Training target values
    ytest             Test target values
    n_cores           Number of CPU cores to use (default: 16)
    n_estimators      Number of trees in the forest (default: 600)
    min_samples_leaf  Minimum samples per leaf node (default: 10)
    min_samples_split Minimum samples per split (default: 2)
    random_state      Random seed for reproducibility (default: 100)

    OUTPUT:
    ----------
    clf      Trained Random Forest model
    ypred    Predicted values for test data
    yhat     Predicted values for training data
    """
    print(f"training RF model with n_estimators = {n_estimators}, min_samples_leaf = {min_samples_leaf}, min_samples_split = {min_samples_split}, and random_state = {random_state}")
    clf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, random_state=random_state, n_jobs=n_cores)
    
    #'fit' implements RF hyperparameters on training dataset
    clf = clf.fit(np.array(xtrain),np.array(ytrain))

    #'predict' takes RF regression and predicts pN2O from input variables
    ypred = clf.predict(np.array(xtest))
    yhat = clf.predict(np.array(xtrain))

    print("Model training complete.")
    print(f'Train Accuracy - : {clf.score(np.array(xtrain),np.array(ytrain)):.3f}')
    print(f'Test Accuracy - : {clf.score(np.array(xtest),np.array(ytest)):.3f}')

    return clf, ypred, yhat


def main():
    """
    DESCRIPTION
    -------------
    Main function to train and evaluate Random Forest models.

    Loads data, trains models on different feature sets, evaluates performance, and saves trained models.
    """
    sns.set_context("paper", rc = {"lines.linewidth":2.5, "font.size": 12,  "font.family": "Arial"})
    datapath, argopath, outputpath, era5path = initialize_paths()
    feature_sets, feature_set_labels, argo_feature_sets = feature_lists()
    go_training, go_test = load_data(datapath, "pN2O")
    
    chosenmodels = [1,2,3,4]
    allplotlabels = [["a", "b", "c"],
                  ["d", "e", "f"],
                  ["g", "h", "i"],
                  ["j", "k", "l"]]

    fig, allaxes = plt.subplots(4,3, figsize = (7.09,10))

    # loop through four models
    for count, modelID in enumerate(chosenmodels):
        feature_list = feature_sets[modelID]
        feature_labels = feature_set_labels[modelID]
        
        # retrain models on full training dataset (don't withold validation data)
        target = ["pN2O"]
        
        xtrain = go_training[feature_list]
        ytrain = np.ravel(go_training[target[0]])
        
        xtest = go_test[feature_list]
        ytest = np.ravel(go_test[target[0]])
        
        # Train the Random Forest model
        clf, ypred, yhat = gen_RF(xtrain, xtest, ytrain, ytest, n_cores=16,
                   n_estimators=600, min_samples_leaf=1, min_samples_split = 2, random_state=100)
  
        # Save the trained model
        dump(clf,f'{outputpath}/model{modelID}_rf_full.joblib')
        print(f"model saved out to {outputpath}/model{modelID}_rf_full.joblib")
        print("feature importances:", clf.feature_importances_)
        
        # Compute and print errors
        print('MAE:', mean_absolute_error(ypred,ytest))
        print('RMSE:', mean_squared_error(ypred,ytest,squared=False))
        
        # Compute R-squared value
        r_square=metrics.r2_score(ytest,ypred)
        print("r square:", r_square)

        # Generate training vs. test plots
        plotlabels = allplotlabels[count]
        axes = allaxes[count]
        plottraintest(fig, axes, plotlabels, go_training, go_test, feature_list, feature_labels, clf, outputpath)

    plt.tight_layout()
    plt.savefig('figures/methods/fullmodeltraintest.png', dpi=300, bbox_inches = "tight")

if __name__ == "__main__":
    main()
