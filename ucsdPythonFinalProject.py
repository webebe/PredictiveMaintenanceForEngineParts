# -*- coding: utf-8 -*-
"""
Created on Tue May 12 06:53:24 2020

@author: benno
"""

#%%Data exploration

#%%Normalisation

#%%traintestsplit

#%%KNN regression

#%%Random forest for regression

#%%LinearRegression

##%Future work for following UCSD Machine Learning course: GLM,GAM, quantile regression, SVM

import pandas as pd
import numpy as np
from numpy import *
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from math import sqrt
#%%Abstract
This report describes the analysis steps conducted on the the Naval Propulsion Plants Data Set available in the UCI machine 
learning repository.
Goal of the analysis is to predict values of the two output variables 'GT Compressor decay state coefficient' and 
'GT turbine decay state coefficient' based on the values of 16 features representing meassurements in the vessel and 
specifically the engine compartment. Basic data exploration steps were conducted, the dataset was cleaned and normalized.
After that the dataset was split into train and test dataset. The machine learning models applied were trained on the training 
set and then evaluated on the test dataset. Results of the analysis can be used to evaluate the state the engine, specifically 
the gas turbine compressor and turbines and to implement condition based or predictive maintanance on the vessel system.
Algorithms applied were linear regression and descision tree regressor. Both algorithms performed well on the given test dataset. 
Decision tree based regression predicted the values of the outcome variables better that linear regression. Accuracy of the 
algorithms was measured by the mean squared error.
#%%Description of the dataset
# A. Coraddu, L. Oneto, A. Ghio, S. Savio, D. Anguita, M. Figari, Machine Learning Approaches for Improving Condition?Based Maintenance of Naval Propulsion Plants, Journal of Engineering for the Maritime Environment, 2014, DOI: 10.1177/1475090214540874, (In Press)
#The experiments have been carried out by means of a numerical simulator of a naval vessel (Frigate) characterized by a Gas Turbine (GT) propulsion plant. The different blocks forming the complete simulator (Propeller, Hull, GT, Gear Box and Controller) have been developed and fine tuned over the year on several similar real propulsion plants. In view of these observations the available data are in agreement with a possible real vessel.
#In this release of the simulator it is also possible to take into account the performance decay over time of the GT components such as GT compressor and turbines.
#The propulsion system behaviour has been described with this parameters:
#- Ship speed (linear function of the lever position lp).
#- Compressor degradation coefficient kMc.
#- Turbine degradation coefficient kMt.
#so that each possible degradation state can be described by a combination of this triple (lp,kMt,kMc).
#The range of decay of compressor and turbine has been sampled with an uniform grid of precision 0.001 so to have a good granularity of representation.
#In particular for the compressor decay state discretization the kMc coefficient has been investigated in the domain [1; 0.95], and the turbine coefficient in the domain [1; 0.975].
#Ship speed has been investigated sampling the range of feasible speed from 3 knots to 27 knots with a granularity of representation equal to tree knots.
#A series of measures (16 features) which indirectly represents of the state of the system subject to performance decay has been acquired and stored in the dataset over the parameter's space.

#Lets have a quick look at the dataset and check out the names and units of the 16 predictor variables and the two output variables
featuredescription = pd.read_csv("C:/Features.txt", header=None)
featuredescription
#%%Read the dataset and check out the variables
data = pd.read_csv("C:/data.txt",sep=" ")
#%%
data.describe()
#%%
data.head()
#%%
data.shape
#%%Remove NA values
data = data.dropna()
#%%
data.isna().any()
#%%no na values in this dataset - nice
#%%Check out variance in of the features
data.var()
#%%gt_comp_inlet_airtemp doesn't have any variance across observations so we remove it because it won't be a good predictor in our models
del data["gt_comp_inlet_airtemp"]
#%%also gt_comp_inlet_airpressure doesn't seem to have a high variance, let's check it out
min(data["gt_comp_inlet_airpressure"])
max(data["gt_comp_inlet_airpressure"])
#%%gt_comp_inlet_airpressure will also be removed due to low variance
del data["gt_comp_inlet_airpressure"]

#%%Set features and target variables
features = ["lever_position","ship_speed","gt_shaft_torque","gt_rateOfrev","gg_rateOfrev","starboard_prop_torque",
            "port_prop_torque","hp_turbine_exittemp","gt_comp_outlet_airtemp","hp_turbine_exitpressure",
            "gt_comp_outlet_airpressure","gt_exhaust_gaspressure","turbine_injection_control",
            "fuel_flow"]
target1=["gt_comp_decay"]
target2=["gt_turbine_decay"]
#%%Scale DF and prepare for Linear regression
X = StandardScaler().fit_transform(data[features])
#%%
y1=data[target1]
y2=data[target2]

#%%Predict target value 1 "GT comp decay" with a linear regression model
X_train, X_test, y1_train, y1_test = train_test_split(X,y1,test_size=0.2,random_state=324)
#%%
regressor=LinearRegression()
regressor.fit(X_train,y1_train)
#%%
y1_prediction=regressor.predict(X_test)
#%%Evaluate Error of prediction
rmse1 = sqrt(mean_squared_error(y_true=y1_test,y_pred=y1_prediction))
print(rmse1)
#%%Predict target value 2 "GT turbine decay" with a linear regression model
X_train, X_test, y2_train, y2_test = train_test_split(X,y2,test_size=0.2,random_state=324)
#%%
regressor=LinearRegression()
regressor.fit(X_train,y2_train)
#%%
y2_prediction=regressor.predict(X_test)
#%%Evaluate Error of prediction
rmse2 = sqrt(mean_squared_error(y_true=y2_test,y_pred=y2_prediction))
print(rmse2)


#%%Predict target value 1  "GT comp decay" with a DecissionTree Regressor
dtregressor1=DecisionTreeRegressor(max_depth=20)
dtregressor1.fit(X_train,y1_train)
#%%
y1dt_prediction=dtregressor1.predict(X_test)
#%%
rmse_DT1 = sqrt(mean_squared_error(y_true=y1_test,y_pred=y1dt_prediction))
print(rmse_DT1)
#%%Predict target value 2  "GT turbine decay" with a DecissionTree Regressor
dtregressor2=DecisionTreeRegressor(max_depth=20)
dtregressor2.fit(X_train,y2_train)
#%%
y2dt_prediction=dtregressor2.predict(X_test)
#%%
rmse_DT2 = sqrt(mean_squared_error(y_true=y2_test,y_pred=y2dt_prediction))
print(rmse_DT2)
#%%
resultsdata = np.array([["RSME Linear Regression GT comp decay","RSME Desc. Tree GT comp decay",
                         "RSME Linear Regression GT turbine decay","RSME Desc. Tree GT turbine decay"],
                        [rmse1,rmse2,rmse_DT1,rmse_DT2]])
#%%Decission tree regressor performes better to predict the GT compressor decay and only a littele better 
#on the turbine decay state.
#%%Which features where most important for the descission tree regressor in order to predict the two outcome variables? 
feat_importances1 = pd.Series(dtregressor1.feature_importances_)
feat_importance1_df = pd.DataFrame(feat_importances1)
feat_importance1_df["feature"]=features
#%%
feat_importances2 = pd.Series(dtregressor2.feature_importances_)
feat_importance2_df = pd.DataFrame(feat_importances2)
feat_importance2_df["feature"]=features
#%%
import matplotlib.pyplot as plt
plt.barh(feat_importance1_df["feature"],feat_importance1_df[0])
plt.title("Feature importance for GT compressor decay")
#%%
plt.barh(feat_importance2_df["feature"],feat_importance2_df[0])
plt.title("Feature importance for GT turbine decay")


##%Future work for following UCSD Machine Learning course: GLM,GAM, quantile regression, SVM
#%%













