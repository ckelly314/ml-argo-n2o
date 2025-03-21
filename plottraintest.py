"""
File: plottraintest.py
----------------------
Created on Weds March 12, 2025

Helper function to plot RF model performance for training
and test data.

@author: Colette Kelly (colette.kelly@whoi.edu)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import metrics

def plottraintest(fig, axes, plotlabels, go_training, go_test, feature_list, feature_labels, RF):

    target_col = "pN2O"

    X_test = np.array(go_test.loc[:,feature_list])
    Y_test = np.array(go_test.loc[:,target_col])
    
    X_train = np.array(go_training.loc[:,feature_list])
    Y_train = np.array(go_training.loc[:,target_col])

    Y_train_pred = RF.predict(X_train)
    Y_test_pred = RF.predict(X_test)

    go_test_SO = go_test[go_test.LATITUDE <= -30]
    train_data_SO = go_training[go_training.LATITUDE <= -30]
    
    X_test_SO = np.array(go_test_SO.loc[:,feature_list])
    Y_test_SO = np.array(go_test_SO.loc[:,target_col])
    
    X_train_SO = np.array(train_data_SO.loc[:,feature_list])
    Y_train_SO = np.array(train_data_SO.loc[:,target_col])
    
    Y_train_pred_SO = RF.predict(X_train_SO)
    Y_test_pred_SO = RF.predict(X_test_SO)

    cmap = plt.get_cmap('ocean')
    colors = cmap(np.linspace(0, 0.5, 3))
    ax = axes[0]
    labels = feature_labels
    ax.bar(labels, RF.feature_importances_*100,
          color = "k", edgecolor = "k")
    ax.set_xticks(range(len(feature_list)))
    ax.set_xticklabels(labels)
    ax.tick_params(direction='in',top=True, right=True)
    ax.set_ylim([0,100])
    ax.set_yticks([0,25,50,75,100])
    ax.set_ylabel('Feature Importance (%)')
    ax.text(-0.25, 1.05, plotlabels[0], fontweight="bold",
                horizontalalignment = "left",
                verticalalignment = "bottom",
                transform = ax.transAxes,
           fontsize = 10)
    
    ax = axes[1]
    ax.set_title("Global")
    ax.tick_params(direction='in',top=True, right=True)
    MAE = mean_absolute_error(Y_test,Y_test_pred)
    RMSE = mean_squared_error(Y_test,Y_test_pred,squared=False)
    r_square=metrics.r2_score(Y_test,Y_test_pred)
    ax.scatter(Y_train, Y_train_pred, label='train',alpha=0.1, color = colors[0])
    ax.scatter(Y_test, Y_test_pred, label='test',alpha=0.1, color = colors[1])
    
    ax.plot(np.linspace(0, 4000), np.linspace(0, 4000),
           color = 'k', zorder = 0, label = "1:1 line")
    ax.set_xlim([0,4000])
    ax.set_ylim([0,4000])
    ax.set_xticks([0,1000,2000,3000,4000])
    ax.set_yticks([0,1000,2000,3000,4000])

    ax.set_xlabel(r"Observed $pN_2O$ (natm)")
    ax.set_ylabel(r"Predicted $pN_2O$ (natm)")
    
    ax.text(0.05, 0.95, 
            f"$R^2$={r_square:.2}\nRMSE={round(RMSE)}\nMAE={round(MAE)}",
           transform = ax.transAxes, verticalalignment = "top",
           fontsize = 10)
    
    ax.text(-0.3, 1.05, plotlabels[1], fontweight="bold",
                horizontalalignment = "left",
                verticalalignment = "bottom",
                transform = ax.transAxes,
           fontsize = 10)

    ax = axes[2]
    ax.set_title("Southern Ocean")
    ax.tick_params(direction='in',top=True, right=True)
    MAE = mean_absolute_error(Y_test_SO,Y_test_pred_SO)
    RMSE = mean_squared_error(Y_test_SO,Y_test_pred_SO,squared=False)
    r_square=metrics.r2_score(Y_test_SO,Y_test_pred_SO)
    ax.scatter(Y_train_SO, Y_train_pred_SO, label='train',alpha=0.1, color = colors[0])
    ax.scatter(Y_test_SO, Y_test_pred_SO, label='test',alpha=0.1, color = colors[1])
    ax.plot(np.linspace(0, 1000), np.linspace(0, 1000),
           color = 'k', zorder = 0, label = "1:1")
    ax.set_xlim([150,850])
    ax.set_ylim([150,850])
    ax.set_xticks([200,400,600,800])
    ax.set_yticks([200,400,600,800])
    ax.legend(loc=4, framealpha = 0)
    ax.set_xlabel(r"Observed $pN_2O$ (natm)")
    ax.set_ylabel(r"Predicted $pN_2O$ (natm)")
    ax.text(0.05, 0.95, 
            f"$R^2$={r_square:.2}\nRMSE={round(RMSE)}\nMAE={round(MAE)}",
           transform = ax.transAxes, verticalalignment = "top",
           fontsize = 10)
    
    ax.text(-0.3, 1.05, plotlabels[2], fontweight="bold",
                horizontalalignment = "left",
                verticalalignment = "bottom",
                transform = ax.transAxes,
           fontsize = 10)

    return fig, axes
