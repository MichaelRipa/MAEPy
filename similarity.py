#! /usr/bin/env python3

__author__ = "Michael Ripa"
__email__ = "michaelripa@cmail.carleton.ca"
__version__ = "1.0.0"
__date__ = "August 26th 2022"

from global_variables import *
from distance import *

import pandas as pd
import numpy as np
from enum import Enum
from math import radians
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import haversine_distances


class Distance(Enum):
    '''Distance - An Enumeration that connects commands to their distance functions, allowing for your choice of distance functions to be implemented very quickly in the pipeline. A brief explanation of the API for the enum values is noted below, but do feel free to build upon this API to fit your needs.

    Enumeration Members:

    EQUALITY : Simple equality, defaults to `_equality()`
    EDIT_DISTANCE : String edit distance, defaults to `_minimal_edit_distance()`
    EUCLIDEAN : For vector valued data, computes Euclidean distance. Defaults to `_euclidean()`
    TIME_DELTA : For computing time differences, defaults to `_time_difference()`

    Values : dict
    A dictionary containing a function and instructions for how to call the function. Current instructions:

    func : distance function
    Distance function to be used

    apply : Bool
    If the provided function does not support vectorized operations (i.e. operations on entire columns of DataFrames or NumPy arrays), then setting `apply` to True makes sure that the dataset is iterated over (less efficient, but necessary in some cases).

    '''
    EQUALITY = {'func': equality,'apply': False}
    EDIT_DISTANCE = {'func':minimal_edit_distance,'apply':True}
    EUCLIDEAN = {'func':euclidean_distance,'apply':False}
    TIME_DELTA = {'func': time_difference,'apply':False}
    JARO_WINKLER = {'func':jaro_winkler_distance,'apply':True}


class Similarity:

    def __init__(self,cols,match_col=match_col,comp_functions=default_comp_functions):
        '''Similarity - A class used for making comparisons between records and generating distance measures to be used for subsequent tasks such as in the training of a classifier or in generating vectorized representations. Stores information about which columns and comparison functions are used, as well as which columns can be used to indicate "gold standard" duplicates.

        Inputs:

        cols : str[]
        A list of columns that will be used in the distance computations

        match_col : str 
        A column that can be used to find known duplicates and non-duplicates

        comp_functions : list of functions
        Optional list of preconfigured functions used to compute distance measures between each of the columns provided. Follows a notation that communicates with the Distance enum, and can be set in `global_variables.py`. 

        '''

        self.cols = cols
        self.comp_functions = comp_functions
        self.match_col = match_col #This is used by parent classes as reference

    def compute_similarities(self,df,compare_df=None):
        '''compute_similarities(self,df)
        Computes the distances (or similarities) between the provided DataFrame(s) in the following 2 possible ways:
        1) Only 1 DataFrame provided (`df`). In this case, every two entries of `df` are compared 
        2) `compare_df` has less entries than `df` (whose length divides into `df`'s length), or equal number of entries. In this case, `compare_df` will be stretched to be the same length as `df` (with repeating entries if smaller) and then both DataFrames are compared. 

        Input:

        df : Pandas DataFrame
        DataFrame of pairs that will have their distances computed. Note that if `compare_df` is not provided, this DataFrame must have pairs placed next to eachother, which is the same format returned when using the `find_pairs` method in preprocessing.

        compare_df : Pandas DataFrame
        DataFrame to compare against `df` entries. Can be smallar than `df` given its length divides `df`'s length.

        Returns:

        n x c NumPy array where n denotes the number of pairs and c denotes the number of columns
        
        '''

        #Case 1: We are given just a single df, pairs are presumed to be adjacent 
        if type(compare_df) == type(None):

            assert len(df) % 2 == 0

            #Grab pairs that will be compared against eachother
            df1 = df[self.cols].iloc[0::2].reset_index(drop=True)
            df2 = df[self.cols].iloc[1::2].reset_index(drop=True)

        #Case 2: We have a DataFrame to compare against
        else:

            #Case 2a: DataFrames same length, we just compare element-wise
            if len(compare_df) == len(df):
                df1 = df[self.cols].reset_index(drop=True)
                df2 = compare_df[self.cols].reset_index(drop=True)

            #Case 2b: One DataFrame shorter than the other : Need to repeat element(s)
            else:
                n = len(df)
                n_compare = len(compare_df)
                #Currently presumes that the comparing df is smallar and fits into df
                assert n % n_compare == 0


                #Repeat the compare_df to match the df size (for efficient vectorized comparison)
                #Note: This is the most efficient approach of repeating a DataFrame that I could find
                df1 = df[self.cols].reset_index(drop=True)
                df2 = compare_df[self.cols].iloc[np.tile([0],n // n_compare)]
                df2.reset_index(drop=True,inplace=True)



        #This approach allows entire columns to have their comparisons computed at once
        return df1.combine(df2, lambda x,y : self.compare(x,y,x.name)).to_numpy()


    def compare(self,s1,s2,col):
        '''compare(self,s1,s2,col)
        Given two Pandas Series objects and a column, uses both the `comp_functions` attribute along with the Distance enumeration to compute the distance of all members of the two Series.

        Inputs:

        s1 : Pandas Series
        Series object containing "left" pairs

        s2 : Pandas Series
        Series object containing "right" pairs

        col : str
        Column in which s1 and s2 came from in DataFrame. Used for looking up enumeration members which allows for the correct functionality from Distance to take place.
        
        Output : Pandas Series
        Returns a Series containing the computed distances of the left and right pairs.


        '''
        
        #This currently assumes that the comp_functions{} dict contains all columns. Will need to make this more explicit

        value = self.comp_functions[col] 
        func = Distance[value].value['func']
        apply = Distance[value].value['apply']
        if apply:
            #Iterates over each element in the Series to compute distance
            return s1.combine(s2, lambda x,y : func(x,y))
        else:
            #Provided function can directly evaluate two Series objects
            return func(s1,s2)

