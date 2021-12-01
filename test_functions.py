"""
Tests for my functions.
"""

from functions import clean_data, plot_feature, perform_regression

import numpy as np
import pandas as pd


def test_clean_data():
    #creating a dataframe with three columns and three rows for testing
    parks_div = pd.DataFrame({'Park Name' : ['A', 'B', 'C'], 'Acres': [1, 2, 3], 'num_species': [2, 7, np.nan]})
    #creating list of column names from previous dataframe I want in dataframe
    get_cols = ['Acres', 'num_species']
    
    #calling clean_data function
    parks_div = clean_data(parks_div, get_cols)
    
    #Test cases -- confirming type and outputs
    assert type(parks_div) == pd.DataFrame
    assert type(get_cols) == list
    assert parks_div.isna().any().sum() == 0
    assert parks_div.shape == (2, 2)

def test_plot_feature():
    #creating a dataframe with three columns and three rows for testing
    parks_div = pd.DataFrame({'Park Name' : ['A', 'B', 'C', 'D', 'E'], 'Acres': [1, 2, 3, 9, 7], 'num_species': [2, 7, 9, 14, 3]})
    #get_cols = ['Park Name', 'Acres', 'num_species']

    #calling the plot_feature function
    fig = plot_feature('scatterplot', 'Acres', parks_div, 'num_species')
    
    #Test cases -- confirming type and valid output
    assert type(parks_div) == pd.DataFrame
    assert fig != np.nan
    

def test_perform_regression():
    #creating a dataframe with three columns and three rows for testing
    parks_div = pd.DataFrame({'Park Name' : ['A', 'B', 'C', 'D', 'E'], 'Acres': [1, 2, 3, 9, 7], 'num_species': [2, 7, 9, 14, 3]})
    
    #calling the perform_regression function
    res = perform_regression(parks_div, 'Acres', 'num_species')
    
    #Test cases -- confirming type and valid output
    assert type(parks_div) == pd.DataFrame
    assert res != np.nan

                 
    