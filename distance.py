#! /usr/bin/env python3

__author__ = "Michael Ripa"
__email__ = "michaelripa@cmail.carleton.ca"
__version__ = "1.0.0"
__date__ = "August 26th 2022"

'''distancy.py - Contains implementations of distance functions to be used in similarity.py

When designing and implementing these distance functions into the pipeline (i.e adding them to the Distance enumeration defined in similarity.py), be mindful of whether the function can support vectorized options between pairs of Pandas Series and Pandas DataFrames. When in doubt, set the `apply` parameter to True when adding a new function to the Distance enmeration.

'''

import numpy as np
import pandas as pd
from math import radians
from jellyfish import levenshtein_distance, jaro_winkler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import haversine_distances


#Definitely feel free to design more distance functions, some untested distance functions have been implemented here to get the ball going! :)

    

def minimal_edit_distance(s1,s2,lower=True):
    '''minimal_edit_distance(s1,s2,lower=True) 
    Computes levenshtein distance with respect to a pair of inputted strigns, checks for pairwise equality in advance. See 

    Inputs:

    s1: str
    First string to be compared

    s2: str
    Second string to be compared

    Output:

    Their edit distance, i.e. the number of insertions, substitutions and deletions needed to turn s1 into s2.

    '''
    
    #This design choice might not be ideal, both in how it treats NaN values, and in the choice of value. A better option may be to look at using a random integer value for missing data or for when one of the strings is NaN.
    try:
        str1 = str(s1)
        str2 = str(s2)
    except:
        return np.inf

    if lower:
        str1 = str1.lower()
        str2 = str2.lower()

    return levenshtein_distance(str1,str2)



def equality(x1,x2):
    '''equality(x1,x2)
    Checks for simple equality between two entries. Note that here, we are treating equality as a distance measure, and so 0 represents an equal pair while 1 represents a different pair.

    Inputs:

    x1 : (any comparable type) 
    First entry being compared

    x2 : (any comparable type) 
    Second entry being compared

    Returns:

    0 : int 
    Indicates pair are a match

    1 : int 
    Indicates pair are not a match

    Note: If passing in Pandas Series or Pandas DataFrames, you need to drop the index before calling this function, i.e:
    x1.reset_index(drop=True)
    x2.reset_index(drop=True)

    '''
    result = (x1 != x2)
    
    if type(result) == bool:
        #Input is a single item
        return 0 if not result else 1
    else:
        #Input is some form of list, one or more items
        return np.array(result,dtype=int)

def euclidean_distance(s1,s2):
    '''euclidean_distance(x1,x2)
    Computes the Euclidean distance of two series of lists.

    Inputs:

    s1 : Pandas Series 
    First series to be compared against. Note entries should be tuples or lists of numbers

    x2 : NumPy Array
    Second vector to be compared against

    Returns:

    Euclidean distance of pair

    '''
    # Turns a list of lists into a 2-dim NumPy array
    v1 = np.stack(s1.values)
    v2 = np.stack(s2.values)

    # See https://numpy.org/doc/stable/reference/ufuncs.html for details behind how vectorized operations like this work.
    return np.sqrt( np.sum( np.square( v1 - v2 ) , axis=1) )
          
def time_difference(t1,t2):
    '''time_difference(t1,t2)
        Computes the distance in time between two inputted datetime objects

        Inputs:

        t1 : NumPY datetime64 object
        First date and time to be compared         

        t2 : NumPY datetime64 object
        Second date and time to be compared         

        Returns:

        Distance in seconds between both dates
    '''
    if type(t1) != np.datetime64 and type(t2) != np.datetime64:
        #Assumption: If t1,t2 are not individual datetimes, then they are either Series or DataFrames of datetime objects
        t1 = t1.values
        t2 = t2.values

    dist = np.abs(t1 - t2)
    return np.divide(dist,np.timedelta64(1,'s')) # Returns time difference in seconds

def mean(X,Y):
    '''mean(X,Y) 
    Returns the mean of two numerical values.

    Inputs:

    X : int* 

    Y : int

    Output:

    Mean of X & Y

    *Note that if X and Y were both NumPy arrays, the returned output would be a vector of elementwise mean comparisons between X and Y. See https://numpy.org/doc/stable/user/basics.broadcasting.html for information  

    '''
    return np.mean([X,Y])


# Left this available for experimentation. This is a distance function specifically for geographical coordinates, incorperates Earth's shape in the calculation.
def haversine_distance(X,Y):
    '''For two coordinates X,Y where both X and Y contain their latitudes and longitudes, returns the Haversine distance'''
    X_radians = [radians(x) for x in X]
    Y_radians = [radians(y) for y in Y]
    # W.L.O.G return distance from X_radians to Y_radians
    return haversine_distances([X_radians,Y_radians])[0,1]

# Added very last minute as an example implementation of a new distance function
def jaro_winkler_distance(s1,s2,lower=True):
    '''jaro_winkler(s1,s2,lower=True) 
    Computes Jaro Winkler distance with respect to a pair of inputted strings, see https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance for mathematical details

    Inputs:

    s1: str
    First string to be compared

    s2: str
    Second string to be compared

    Output:

    Jaro-Winkler distance of the pair of strings.

    '''
    
    try:
        str1 = str(s1)
        str2 = str(s2)
    except:
        return np.inf

    if lower:
        str1 = str1.lower()
        str2 = str2.lower()

    return 1 - jaro_winkler(str1,str2)

