"""A collection of functions for doing my project."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import patsy
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import ttest_ind, chisquare, normaltest

#what does the pass inside of each function do?

def clean_data(df, col_names):
    '''
    will clean the input dataframe by dropping all rows with null values,
    and only keeping the data from columns that will be used for analysis
    
    Parameters
    -----------------
    df : df
        dataframe to be cleaned
        
    Returns
    -----------------
    new_df : df
        dataframe that is cleaned (has null rows dropped and only contains
        significant columns)
    '''
    
    #will filter the dataframe to contain all column names in list col_names
    new_df = df[col_names]

    #if any of the values in any of the columns are null, then will drop all rows
    #that contain null values
    if new_df.isna().any().sum() >= 1:
        new_df = new_df.dropna(subset = col_names)
        return new_df
    else:
        return new_df

def plot_feature(kind, x, data, y = None):
    '''
    will plot the input feature based on kind of plot that is specified
    
    Parameters
    -----------------
    kind : string
        the kind of plot that will be created
    x : str
        the feature that will be plotted
    y : str or None
        None if there is no second feature that will be plotted; str if there is second feature
        that will be plotted
    data : df
        dataframe that contains feature data
        
    Returns
    -----------------
    plot : fig
        the plot of the feature data
    '''
    
    #Will plot a histogram of feature x from data
    if kind == 'histogram':
        plot = sns.histplot(x = x, data = data)
        return plot
    #Will plot a scatterplot of features x vs y from data
    elif kind == 'scatterplot':
        plot = sns.scatterplot(x = x, y = y, data = data)
        return plot

def perform_regression(df, x, y):
    '''
    will perform a linear model regression analysis on two specified features x and y
    
    Parameters
    -----------------
    df : df
        dataframe that contains feature data
    x : str
        feature number one
    y : str
        feature number two
        
    Returns
    -----------------
    res : statsmodels
        fitted model from the regression analysis that was performed using patsy   
    '''
    
    #initialize the regression model
    outcome, predictors = patsy.dmatrices(x + ' ~ ' + y, df)
    mod = sm.OLS(outcome, predictors)
    #fit the regression model
    res = mod.fit()
    
    return res