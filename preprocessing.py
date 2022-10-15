#! /usr/bin/env python3

__author__ = "Michael Ripa"
__email__ = "michaelripa@cmail.carleton.ca"
__version__ = "1.0.0"
__date__ = "August 26th 2022"

from global_variables import * 

import re 
import pgeocode
import numpy as np
import pandas as pd
from copy import deepcopy
from enum import Enum
from scipy.special import comb

# Used for gaining additional geographical information from (canadian) postalcodes
nomi = pgeocode.Nominatim('CA')
postal_regex = re.compile('[A-Za-z][0-9][A-Za-z][0-9][A-Za-z][0-9]')

def load_df(path,default_preprocessing=True,nrows=None):
    '''load_df(path,default_preprocessing=True,nrows=None)
    Loads data provided from a path to a csv or parquet file into a Pandas DataFrame object, and allows for additional preprocessing to take place.

    Inputs:

    path : str
    Path to csv or parquet file

    default_preprocessing : Bool
    Allows for preprocessing to take place if inputted data has not been cleaned

    nrows : int
    Allows for a subset of the data to be loaded in. Defaults to the entire dataset

    Returns:

    df : Pandas DataFrame

    '''
    assert type(path) == str
    if path[-4:] == '.csv':
        df = pd.read_csv(path,engine='python',nrows=nrows)
    elif '.parquet' in path:
        df = pd.read_parquet(path,engine='auto')
        if type(nrows) == int:
            #This is really bad practice, but wanted to avoid external dependancies that would be necessary to allow for this operation to take place at read time.
            df = df.iloc[nrows]

    if default_preprocessing:
        return preprocess_df(df)
        
    return df

#This function is really not ideal, should be developed further for your own usecases.
def preprocess_df(df,geodata=True,datetimes=True):
    '''preprocess_df(df,geodata=True,datetimes=True)
       A helper function that can be used to apply some basic preprocessing to a provided DataFrame. Note that this function is poorly designed, and should not be relied on outside of the context of PSRS Graduates. Ideally, experimental data should be preprocessed once and stored for reuse, this just exists for situations where a new dataset needs to be processed. Note that this function can be called automatically from `load_df()`.

        Inputs:

        df : Pandas DataFrame
        Dataset to be preprocessed on. 
            

        geodata : Bool
        If True, tries to add geographical data by querying postalcodes with the pgeocode library, and also merges the returned latitude & longitude columns into one (named lat-long). Otherwise, skips the step altogether. If geodata exists, performs extra preprocessing step.

        datetimes : Bool
        If True, turns elements of columns in `DATETIME_COLS` into datetime64 objects. The choice of columns (i.e. the `DATETIME_COLS` value) can be set in global_variables.py

        Returns:

        df : Pandas DataFrame
        Preprocessed DataFrame

    '''


    #This adds geographical data from postalcodes and merges the latitude and longitude together. Note that you can save a dataset with that data populated and skip this step if wanting to increase performance.
    if geodata:
        #Data here is left hardcoded (not good) but should be consistent as it comes from pygeocode.
        if 'accuracy' not in df.columns:
            df = add_geodata_from_postalcodes(df)

        if 'lat-long' not in df.columns:
            latlong_cols = ['latitude','longitude']
            # Extra preprocessing step to avoid crashing from lists containing NaN. Fills NaN latitude and longitude values randomly (uniform w.r.t max and min value)
            for col in latlong_cols:
                df.loc[:,col] = replace_rand(df.loc[:,col],uniform=True)
            df = merge_columns(df,cols=latlong_cols,merge_col='lat-long',drop_old_cols=True)

        df['accuracy'] = df['accuracy'].fillna(0)

    if datetimes:
        df = convert_df_to_datetime(df,DATETIME_COLS)

    #Left hardcoded as operation is specific only to PRI. Remove in different usecases.
    df[(df['PRI'] == 00000000) | (df['PRI'] == 0)]['PRI'] = np.nan

    return df

def replace_rand(s,uniform=False):
    '''replace_rand(s,uniform=False)
    For a given Pandas Series, replaces nan values with randomly selected values based on the distribution of the data.

    Inputs:

    s : Pandas Series
    Pandas Series to be cleaned

    uniform : Bool
    If True, then the distribution is uniform, otherwise a random value from the non null data is selected 

    Returns:
    s : Pandas Series
    Cleaned Series with all nan values filled in w.r.t `uniform`

    '''
    mask = s.isna()
    #Draws random sample from uniform distribution (w.r.t min/max of the series)
    if uniform:
        high = s.max()
        low = s.min()
        s.loc[mask] = np.random.uniform(low=low,high=high,size=np.sum(mask))
        return s

    #Draws random sample from actual distribution of the underlying series.
    distribution = s[~mask].values
    s[mask] = np.random.choice(distribution,np.sum(mask))
    return s

   
def add_geodata_from_postalcodes(df,postal_code_col=postal_code):
    '''add_geodata_from_postalcodes(df,postal_code_col=postal_code)
    Uses the pgeocode API to generate more geographical information with respect to a provided column of postalcodes of a provided dataset. See https://pgeocode.readthedocs.io/en/latest/generated/pgeocode.Nominatim.html for additional information behind the pgeocode library.

    Inputs:

    df : Pandas DataFrame
    DataFrame to have geographical data generated for.

    postal_code_col : str
    Column of `df` cooresponding to postalcode data. Defaults to `postal_code` which is set in global_variable.py

    Returns:

    df : Pandas DataFrame
    Original DataFrame with new geographical columns and data added to it.

    '''
    #TODO: At this time, the only postalcodes that are processed are Canadian ones. The pgeocode API supports different countries, and based on the PSRS Graduates table, it seems feasible to create support for certain other countries.


    # The pgeocode API throws an error when empty strings are passed in, hence the need for sanitation.
    df.loc[df[postal_code_col] == '',postal_code_col] = '0'
    # See https://pandas.pydata.org/docs/user_guide/text.html for details on vectorized string operations     
    query = df.loc[:,postal_code_col].str.lower().str[0:3] + ' ' + df.loc[:,postal_code_col].str.lower().str[3:] 
    postal_df = nomi.query_postal_code(query.values)
    df[postal_df.columns] = postal_df
    return df

def convert_df_to_datetime(df,cols):
    try:
        converted_df = pd.to_datetime(df[cols])
    except:
        raise ValueError('One or more columns not able to be converted to type datetime: ' +str(cols))

    df[cols] = converted_df
    return df

def merge_columns(df,cols,merge_col,drop_old_cols=True):
    '''merge_columns(df,cols,merge_col,drop_old_cols=True)
    Helper function used to merge two or more columns into a list using computationally efficient vectorization methods.

    Inputs:

    df : Pandas DataFrame
    DataFrame to have column merge applied to

    cols : str[]
    List of columns to be merged together

    merge_col : str
    Name of new column containing merged values

    drop_old_cols : Bool
    Once merged, drops the original columns from the DataFrame

    Returns:

    df : Pandas DataFrame
    Original DataFrame with columns merged to `merge_col`, and (optionally) with the old columns removed.


    '''

    df.loc[:,merge_col] = df.loc[:,cols].values.tolist()

    if drop_old_cols:
        return df.drop(cols,axis=1)

    return df
    
def find_pairs(df,similarity,n=None,duplicates=True,strict_pair=True,move_to_top=False,return_ratio=True,subset_cols=True,additional_cols=None):
    '''From a given dataset, finds pairs of duplicates or non-duplicates with respect to the match column of the provided similarity instance.

    Inputs:

    df : Pandas DataFrame
    Dataset to draw pairs from. 

    similarity : Similarity
    Similarity instance used to determine the match column as well as the columns to return (if specified)

    n : int
    Number of pairs to return. Defaults to returning all possible pairs

    duplicates : Bool
    If True, finds pairs with identical match column values, otherwise returns pairs with differing (non-null) match column values.

    strict_pair : Bool
    In certain circumstances, it only makes sence to find duplicate pairs (such as when move_to_top is set true). Setting this as True forces pairs to have both records unique.

    move_to_top : Bool
    If True, instead of just returning the determined pairs, they are placed at the top of the dataset and the whole dataset is returned. Useful within the MAEPy pipeline.

    return_ratio : Bool
    If True, returns the ratio of the duplicate pairs.

    subset_cols : Bool
    If True, only keeps the columns specified in `similarity`, as well as the match column. Reduces the amount of memory needed for the function, as well as for what the data is used for down the line.

    Returns:

    pairs_df : Pandas DataFrame
    DataFrame containing pairs in every second entry. If move_to_top set, this DataFrame will also contain the remaining data below the pairs.
    
    '''
    match_col = similarity.match_col
    # Only grab columns used for comparison in similarity, along with the match column and any additionally specified columns.
    if subset_cols:
        cols = similarity.cols
        col_subset = cols + [match_col]

        if type(additional_cols) != type(None):

            #If this error is thrown, just place the column(s) into a python list.
            assert type(additional_cols) == list
            col_subset += additional_cols
            # Remove duplicate columns
            col_subset = list(np.unique(col_subset))

            
            
        df = df.loc[:,col_subset]

        

    counts = df[match_col].dropna().value_counts()
    
    if duplicates:
        
        # Grab all duplicates w.r.t `match_col` given the value of `match_col` is non-null.
        #duplicates_df = df[df[match_col].duplicated(keep=False) & df[match_col].notna()]

        #Start by only selecting strict pairs
        strict_pairs_indices = counts[counts == 2].index
        pairs_df = df[df[match_col].isin(strict_pairs_indices)]

        pairs_df = pairs_df.sort_values(match_col)
        # Create pairs for larger duplicate groups
        if not strict_pair:
            for num_dups in counts:
                if num_dups > 2:
                    dup_group = counts[counts == num_dups].index
                    for cur_match in dup_group:
                        dup_group_indices = df[df[match_col].isin([cur_match])].index
                        pair_indices = compute_all_pairs(dup_group_indices)
                        pairs_df = pd.concat([pairs_df,df.iloc[pair_indices]])
                        
            
    else:
        # Keep records that have a unique, non-null `match_col` value
        if strict_pair:
            pairs_df = df[df[match_col].isin(counts[counts == 1].index)]

        # Generates all possible pairs of these unique records
        else:
            non_duplicate_indices = compute_all_pairs(counts[counts == 1].index)
            pairs_df = df[df[match_col].isin(non_duplicate_indices)]
            
            #df = df[(~df[join_col].duplicated(keep=False)) | df[join_col].isna()]

    total_pairs = ( len(pairs_df) // 2 )
    if n != None:
        #Ensure that there are enough available pairs
        assert n <= total_pairs

    else:
        n = total_pairs

    
    if duplicates:
        pairs_df = check_transposition(pairs_df)
        pairs_df = shuffle_pairwise(pairs_df)
    #Return dataset w.r.t specifications of `move_to_top` and `return_ratio`
    ratio = 2*n / len(df)
    if not move_to_top:
        if return_ratio:
            return pairs_df.iloc[0: 2*n] , ratio
        return pairs_df.iloc[0: 2*n]
        
    indices = list(set(pairs_df.iloc[0:2*n].index))
    non_pairs_df = df[~df.index.isin(indices)]
    if return_ratio:
        return pd.concat([pairs_df.iloc[0:2*n],non_pairs_df]) , ratio

    return pd.concat([pairs_df,non_pairs_df])


def check_transposition(given_pairs,cols=transposed_pair,print_results=False):
    '''check_transposition(given_pairs,cols,print_results=False)
    Given a pair of matches and columns, checks whether there is a transposition
    
    Inputs:

    given_pairs : Pandas DataFrame
    DataFrame of pairs. Assumed that every two entries are a pair to be compared against

    cols : str
    Pair of columns to check for transpositions on. Defaults to what is set in `transposed_pair` in global_variables.py

    print_results : Bool

    If trying to see examples of transposed pairs, setting this to True prints these pairs to the screen.

    Returns:

    pairs : Pandas DataFrame
    DataFrame with transposed pairs fixed (swapped to the same order).
    '''

    #Not too proud of the readability of this one, a vectorized approach could definitely be implemented (i.e. checking for transpositions over entire df) but ran out of time.
    pairs = deepcopy(given_pairs)
    c1 = cols[0]
    c2 = cols[1]
    for i in range(len(pairs)//2):
        pair = pairs.iloc[2*i: 2*i + 2]
        col1_val1 = pair[c1].iloc[0].lower()
        col1_val2 = pair[c1].iloc[1].lower()
        col2_val1 = pair[c2].iloc[0].lower()
        col2_val2 = pair[c2].iloc[1].lower()
        if col1_val1 == col2_val2 and col1_val2 == col2_val1:
            if print_results:
                print(pair[cols])
            pairs.iloc[2*i, pairs.columns.get_loc(c1)] = col1_val2
            pairs.iloc[2*i, pairs.columns.get_loc(c2)] = col2_val2


    return pairs


def shuffle_pairwise(df):
    '''shuffle_pairwise(df)
    Shuffles a DataFrame in such a way that each entry in index 2i and 2i +1 remain neighbours. In other words, if 2i goes to 2j, then 2i + 1 goes to 2j + 1

    Input:

    df : Pandas DataFrame
    Dataset to be shuffled. Should have every two rows cooresponding to a duplicate (or pair that you want to be kept together).
    
    Returns:

    df : Pandas DataFrame
    Shuffled version of original DataFrame (with pairs kept adjacent)
    '''
    assert len(df.index) % 2 == 0
    n_pairs = len(df.index) // 2
    permutation = np.arange(n_pairs)
    np.random.shuffle(permutation)
    df.iloc[0::2] = df.iloc[2*permutation]
    df.iloc[1::2] = df.iloc[2*permutation + 1]
    return df

def create_training_labels(X_true,X_false):
    '''create_training_labels(X_true,X_false)
    Takes in two NumPy arrays of true pairs and false pairs and creates a shuffled labeled training set. Used for creating training data for supervised learning.
    
    Inputs:

    X_true : NumPy Array
    Array corresponding to true values (labeled as "1")

    X_false : NumPy Array
    Array corresponding to false values (labeled as "0")

    Returns:

    XY : NumPy Array
    Array with combined X_true, X_false arrays, shuffled and labeled.

    '''
    y = np.concatenate([np.ones((len(X_true),1)),np.zeros((len(X_false),1))])
    X = np.concatenate([X_true,X_false])
    XY = np.random.permutation(np.concatenate([X,y],axis=1))
    return XY[:,0:-1] , XY[:,-1]

def compute_all_pairs(index,combine=True):
    '''compute_all_pairs(index,combine=True)
    For an inputted list of integers (cooresponding to indicies of a DataFrame), returns all possible pairs of these indices. Extremely useful in both creating more training data for records consisting of > 2 duplicates and in comparing amongst clusters in Dedupe.

    Inputs:

    index : int[]
    NumPy array of indices (of some Pandas DataFrame)

    combine : Bool
    If True, returns [a,b,c] as [a,b,a,c,b,c] *
    If False, returns [a,b,c] as [a,a,b] , [b,c,c] *

    *as NumPy arrays

    '''
    index = np.array(index)
    n_indices = len(index)

    assert n_indices > 1
    
    n_comparisons = int(comb(n_indices,2))

    # Explanation: The goal here is to take a list of labels [a,b,c,d] and find a way to represent all possible pairs: [a,b] , [a,c] , [a,d] , [b,c] , [b,d] , [c,d]

    # This step would take [a,b,c,d] and obtain [a,a,a,b,b,c]
    left_index_values = np.repeat(index,np.arange(n_indices-1,-1,-1,dtype=np.int64))

    #This step would take [a,b,c,d] and obtain [b,c,d,c,d,d]
    
    right_index_values = np.array([index[i] for j in range(n_indices) for i in range(j+1,n_indices)])

    #Return pairs in desired format
    if combine:
        return np.array([left_index_values,right_index_values]).flatten(order='F')

    return left_index_values , right_index_values


