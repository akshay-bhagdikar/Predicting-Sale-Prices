import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns


#setting the working directory
os.chdir('C:/Users/aksha/Desktop/Machine learning/Housing project')

#Importing the training dataset 
dataset_train = pd.read_csv('housing_train.csv')

#Importing the testing dataset
dataset_test = pd.read_csv('housing_test.csv')

#preview of the data
dataset_train.head()

#overall info of the dataset
dataset_train.info()

#separating the dependent and independent variables
X_train = dataset_train.iloc[:,1:80].values
y_train = dataset_train.iloc[:,80:81].values  
X_test = dataset_test.iloc[:,1:80].values
                            

#check for missing values 
percentage_null = np.zeros((X_train.shape[1],1))
for i in range(0,X_train.shape[1]):
    inds_null = len((np.where(pd.isnull(X_train[:,i])))[0])
    percentage_null[i] = inds_null/X_train.shape[0]*100

#deleting alley,fireplace,poolQc, fence, MiscFeatures columns due to huge missing data and less relevance
X_train = np.delete(X_train, [5,56,71,72,73], axis=1)
X_test = np.delete(X_test, [5,56,71,72,73], axis=1)

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:, 2:3])
X_train[:, 2:3] = imputer.transform(X_train[:, 2:3])
X_test[:,2:3] = imputer.transform(X_test[:,2:3])
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:,24:25])
X_train[:,24:25] = imputer.transform(X_train[:,24:25])
X_test[:,24:25] = imputer.transform(X_test[:,24:25])


#correlation matrix
corrmat = dataset_train.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

#deleting garage year built column as it is highly collinear with year built and both variables give almost the same information
X_train = np.delete(X_train, [56], axis=1)
X_test = np.delete(X_test, [56], axis=1)

#deleting TotRmsAbvGrd column as it is highly collinear with GrLivArea and both variables give almost the same information
X_train = np.delete(X_train, [52], axis=1)
X_test = np.delete(X_test, [52], axis=1)

#deleting TotalBsmtSF column as it is highly collinear with 1stFlrSF 
X_train = np.delete(X_train, [36], axis=1)
X_test = np.delete(X_test, [36], axis=1)

#deleting garage area column as it is highly collinear with garage cars and both the variables give almost the same information
X_train = np.delete(X_train, [56], axis=1)
X_test = np.delete(X_test, [56], axis=1)

#deleting rows with missing values
for i in range(0,X_train.shape[1]):
    inds_null = np.where(pd.isnull(X_train[:,i]))[0]
    X_train = np.delete(X_train,inds_null,axis=0)
    y_train = np.delete(y_train,inds_null,axis=0)

for i in range(0,X_test.shape[1]):
    inds_null = np.where(pd.isnull(X_test[:,i]))[0]
    X_test = np.delete(X_test,inds_null,axis=0)
    
#Separating categorical variables(cardinal in nature)
X_train_cat  = []
X_train_ncat = []
X_test_cat   = []
X_test_ncat  = []

for i in range(0,X_train.shape[1]):
    if i!=2 and i!=3 and i!=15 and i!=16 and i!=17 and i!=18 and i!=24 and i!= 32 and i!=34 and i!=35 and i!=40 and i!=41 and i!= 42 and i!=43 and i!=44 and i!=45 and i!= 46 and i!=47 and i!=48 and i!=49 and i!=52 and i!=55 and i!=59 and i!=60 and i!=61 and i!=62 and i!=63 and i!= 64 and i!=65 and i!= 66 and i!=67:
        X_train_cat = np.append(X_train_cat,X_train[:,i])
        X_test_cat = np.append(X_test_cat,X_test[:,i])
    else:
        X_train_ncat = np.append(X_train_ncat,X_train[:,i])
        X_test_ncat = np.append(X_test_ncat,X_test[:,i])
        
X_train_cat = np.transpose(X_train_cat.reshape(int(X_train_cat.shape[0]/X_train.shape[0]),X_train.shape[0])) 
X_train_ncat = np.transpose(X_train_ncat.reshape(int(X_train_ncat.shape[0]/X_train.shape[0]),X_train.shape[0])) 
X_test_cat = np.transpose(X_test_cat.reshape(int(X_test_cat.shape[0]/X_test.shape[0]),X_test.shape[0])) 
X_test_ncat = np.transpose(X_test_ncat.reshape(int(X_test_ncat.shape[0]/X_test.shape[0]),X_test.shape[0])) 

#plotting non categorical variables vs sale price
for fig in range(0,X_train_ncat.shape[1]):
    plt.scatter(X_train_ncat[:,fig],y_train)
    plt.show()
    
#deleting enclosed porch, 3sn porch, Screen porch, pool area, misc value column as it does not give much information (most of the values are zero)
inds = [28,27,26,25,24]
X_train_ncat = np.delete(X_train_ncat, inds, axis=1)
X_test_ncat = np.delete(X_test_ncat, inds, axis=1)


#labeling the categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
ind_del=[]
for i in range(0,X_train_cat.shape[1]):
    try:
        labelencoder = LabelEncoder()
        X_train_cat[:,i] = labelencoder.fit_transform(X_train_cat[:,i])
        X_test_cat[:,i] = labelencoder.transform(X_test_cat[:,i]) 
        
    except:
        ind_del = np.append(ind_del,i)
        continue
    X_train_cat = np.delete(X_train_cat,ind_del,axis=1)
    X_test_cat = np.delete(X_test_cat,ind_del,axis=1)
        
onehotencoder = OneHotEncoder(categorical_features='all')
X_train_cat = onehotencoder.fit_transform(X_train_cat).toarray()
X_test_cat = onehotencoder.transform(X_test_cat).toarray()


#removing the outliers
for j in range(0,X_train_ncat.shape[1]):
    col = np.sort([X_train_ncat[:,j]])
    if col.shape[1]%2 == 0:
        med_ind = int(col.shape[1]/2)
    else: 
        med_ind = int((col.shape[1]+1)/2)
    col_upper = col[0,0:med_ind]
    col_lower = col[0,med_ind:col.shape[1]]
    
    quartile_1 = np.median(col_upper)
    quartile_2 = np.median(col)
    quartile_3 = np.median(col_lower)
    diff = 3*(quartile_3-quartile_1)
    fence_up = quartile_3 + diff
    fence_low = quartile_1 - diff
    inds_low = np.where(X_train_ncat[:,j] < fence_low)
    inds_up = np.where(X_train_ncat[:,j] > fence_up)
    inds = np.append(inds_low[0],inds_up[0])
    X_train_ncat = np.delete(X_train_ncat, inds, axis = 0)
    X_train_cat = np.delete(X_train_cat, inds, axis = 0)
    y_train = np.delete(y_train, inds)

X_net_train = np.concatenate((X_train_ncat, X_train_cat), axis=1)

for j in range(0,X_test_ncat.shape[1]):
    col = np.sort([X_test_ncat[:,j]])
    if col.shape[1]%2 == 0:
        med_ind = int(col.shape[1]/2)
    else: 
        med_ind = int((col.shape[1]+1)/2)
    col_upper = col[0,0:med_ind]
    col_lower = col[0,med_ind:col.shape[1]]
    
    quartile_1 = np.median(col_upper)
    quartile_2 = np.median(col)
    quartile_3 = np.median(col_lower)
    diff = 3*(quartile_3-quartile_1)
    fence_up = quartile_3 + diff
    fence_low = quartile_1 - diff
    inds_low = np.where(X_test_ncat[:,j] < fence_low)
    inds_up = np.where(X_test_ncat[:,j] > fence_up)
    inds = np.append(inds_low[0],inds_up[0])
    X_test_ncat = np.delete(X_test_ncat, inds, axis = 0)
    X_test_cat = np.delete(X_test_cat, inds, axis = 0)
    y_test = np.delete(y_test, inds)

X_net_test = np.concatenate((X_test_ncat, X_test_cat), axis=1)


#standardizing the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_net_train = sc_X.fit_transform(X_net_train)
X_net_test = sc_X.transform(X_net_test)
sc_y = StandardScaler()
y_train_scaled = sc_y.fit_transform(y_train)


#implementing PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 150) 
X_net_train = pca.fit_transform(X_net_train)
X_net_test = pca.transform(X_net_test)
explained_variance = pca.explained_variance_ratio_


#Implementing linear regression
from sklearn.linear_model import LinearRegression
regressor_linreg = LinearRegression()
regressor_linreg.fit(X_net_train, y_train_scaled)
y_pred_linreg = regressor_linreg.predict(X_net_test)       #predicting the sale prices
y_scaled_linreg = sc_y.inverse_transform(y_pred_linreg)

#implementing SVM
from sklearn.svm import SVR
regressor_svr = SVR(kernel = 'linear')
regressor_svr.fit(X_net_train,y_train_scaled)
y_pred_svr = regressor_svr.predict(X_net_test)
y_scaled_svr = sc_y.inverse_transform(y_pred_svr)

#implementing Decision tree regression
from sklearn.tree import DecisionTreeRegressor
regressor_dtr = DecisionTreeRegressor(random_state = 0)
regressor_dtr.fit(X_net_train,y_train_scaled)
y_pred_dtr = regressor_dtr.predict(X_net_test)
y_scaled_dtr = sc_y.inverse_transform(y_pred_dtr)

#implementing Random forest regression
from sklearn.ensemble import RandomForestRegressor
regressor_RFR = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor_RFR.fit(X_net_train,y_train_scaled)
y_pred_RFR = regressor_RFR.predict(X_net_test)
y_scaled_RFR = sc_y.inverse_transform(y_pred_RFR)

#analysing linear regressor performance
import statsmodels.formula.api as sm
X_net_train = np.append(arr = np.ones((X_net_train.shape[0],1)).astype(int), values = X_net_train, axis = 1)
ncol = np.arange(X_net_train.shape[1])
X_opt = X_net_train[:,ncol]
regressor_ols = sm.OLS(endog = y_train_scaled, exog = X_opt).fit()
regressor_ols.summary()




        

            
        
            
            
            












