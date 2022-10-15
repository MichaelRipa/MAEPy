#! /usr/bin/env python3

__author__ = "Michael Ripa"
__email__ = "michaelripa@cmail.carleton.ca"
__version__ = "1.0.0"
__date__ = "August 26th 2022"

from global_variables import *
from similarity import Similarity

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Grad2Vec:

    def __init__(self,similarity,records):
        '''Grad2Vec: A class for converting non-numerical records (traditionally PSRS graduates) into numerical vectors in which similar records have a lower Euclidean distance (with respect to the embedding).

        Inputs:

        similarity : Similarity()
        Similarity instance used in aiding with the creation of the embeddings.

        records : df
        Pandas DataFrame containing records to be embedded as vectors


        Resulting attributes:
     
        similarity : Similarity()

        records : df


        '''

        self.records = records
        self._skew_embeddings = False 
        self.similarity = similarity

    def generate_embeddings(self, k=10, splits=10,default_basis=None,apply_changes=True,apply_mean=True):
        '''generate_embeddings(self, k=10, splits=10,default_basis=None,apply_changes=True,apply_mean=True)
        Creates a vectorized representation (or embedding) for an inputted DataFrame of records with respect to a random sample of the data.

        Inputs:

        k : int 
        Size of sample used for creating embeddings 

        splits : int 
        How many groups of the sample to be combined

        default_basis : Pandas DataFrame 
        Sample to be used for creating embedding (optional)

        apply_changes : Bool 
        If True, updates the model with the new embeddings and meta info

        apply_mean : Bool
        If k != splits, there will be a combination of distance vectors performed. Setting `apply_mean` as True takes the mean value, while False just combines them via summation

        Output:

        embeddings : NumPy Array
        NumPy array of dimension n_records x (n_cols x n_splits)


        Resulting Attributes:

        n_records : int

        k : int

        splits : int

        basis : Pandas DataFrame

        embeddings : Numpy array 
        NumPy array of dimension n_records x (n_cols x n_splits)

        '''
    
        assert k % splits == 0

        n_records = len(self.records.index)
        basis = default_basis
        m = len(self.similarity.cols)
        
        # Obtain basis if not provided
        if type(basis) == type(None):
            basis = self._get_basis(k)
        embeddings = np.zeros((n_records,m*splits))
        cur_split = 0

        #Compute distance to each "basis element" and add to slot in embedding (potentially adding to 1 or more pre-existing distances)
        for i in range(k):
            vector = basis.iloc[i:i+1,:]
            embeddings[:,cur_split*m : (1 + cur_split)*m] += self.similarity.compute_similarities(self.records,vector)
            cur_split = (cur_split + 1) % splits 

        #This takes the mean distance rather than the sum of distances for the embedding
        if apply_mean:
            embeddings = np.divide(embeddings,k // splits)

        if apply_changes:
            self.n_records = n_records 
            self.k = k
            self.splits = splits
            self.basis = basis
            self.embeddings = embeddings
            return self
        return embeddings


    def set_classifier_weights(self,weights):
        '''set_classifier_weights(self,weights)
        To improve the accuracy of the weights, one can train a linear classifier (like Logistic Regression) on pairwise duplicates and use the training weights to skew the vectorspace with respect to how important a dimension is in determining similarity. Note that in order to preserve the raw embeddings, these weights are left seperate from the embeddings, left for a parent class to add to the embeddings when used in computations. In other words, `embeddings` will never be modified at any point.

        Inputs:

        weights : float[]
        NumPy array of weights

        Resulting Attributes:        

        weights : float[]

        '''
        self.weights = weights
        self._skew_embeddings = True

    #This function definitely has room for a more creative approach, such as selecting a sample of very distinct records to use as a "basis" instead of totally random.
    def _get_basis(self,k):
        '''_get_basis(self,k)
        Helper function which grabs "basis" i.e. a sample of k records from `records`, all of which have non-null values. This "basis" gets used in creating the embeddings (every record has its distance taken with each basis element).
        '''

        cols = self.similarity.cols
        non_null = self.records[cols].notna().all(axis=1)
        index = np.random.choice(np.arange(0,np.sum(non_null)),size=k)
        return self.records[non_null][cols].iloc[index]
    
