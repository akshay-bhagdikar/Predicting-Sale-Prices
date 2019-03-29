
## Introduction:

Machine learning is not only about applying machine learning algorithms to your data but it is also equally about pre processing the data and making it ready for machine learning algorithms to work on it. Machine learning algorithms are finicky in the sense that they require their data in specific condition. For example, linear regression (and many other algorithms) require the data to be in numeric format. They can not work with categorical variables such as names. Apart from the format of data, the purity of data is also an important concern. By purity I mean whether data has lot of null values or outliers in it. These are some of the concerns that we need to address before we start applying machine learning algorithms to the data. 
In this project I try to explore data pre processing and machine learning techniques for regression. The dataset used here is housing dataset from kaggle (https://www.kaggle.com/c/house-prices-advanced-regression-techniques). The aim of this project is to be able to predict the pricing of a house given some of its feature.
****************************************************************************************************************************************
## About the dataset:

The raw data consists of 79 explanatory features such as  Lot Area, Neighborhood, Type of dwelling, Year Built and so on. It has 1460 rows. The description of the features can be found in the data_description.txt file.

********************************************************************************************************************************************************************************************************************************************************************************

## Data pre-processing:

This step helps us to know about the data distribution, missing values, outliers etc. 

**Missing data**

In our case the columns - Alley, FirePlaceQu, PoolQc, Fence, MiscFeature have huge amount of missing values (more than 47% of the data) and the occurences of their values missing do not seem to depend on other variables. So we can safely delete those columns. 
I am also deleting the id column as it does not provide any useful information. For other columns with missing data I imputed the values with some appropriate values. 

**Dealing with outliers!**

I plotted the pair plots (scatter plots for every variable against every other variable) and histograms to look out for the distribution of the data and finding out the outliers. Outliers can be seen sitting at some corner of the scatter plot or in the tail regions of the histograms. 

********************************************************************************************************************************************************************************************************************************************************************************

## Feature Selection

The next section is about feature selection. I have used two methods of feature selection - 
- Based on  visual clues - By looking at correlation heatmap and boxplots
- Based on statistical parameters - By measuring the p values


********************************************************************************************************************************************************************************************************************************************************************************

## Machine Learning Models 

I applied linear regression, decision tree, random forest and SVM models to the data. I evaluated their performance by comparing the r2 values.


********************************************************************************************************************************************************************************************************************************************************************************

## Summary:

In this project we discussed handling missing values, dealing with ouliers, feature selection, different machine learning models and evaluating their performance based on R2 value. 
We found that by basing the feature selection on manual selection (visual means) we select around 14 features and by basing it on statistical measures we select 55 features. By selecting more features we get an increase in r2 score by 0.3. 
Out of all the machine learning models that we applied, we got better performance by using decision tree model.!!!!!!

********************************************************************************************************************************************************************************************************************************************************************************
