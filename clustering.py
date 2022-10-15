#! /usr/bin/env python3

__author__ = "Michael Ripa"
__email__ = "michaelripa@cmail.carleton.ca"
__version__ = "1.0.0"
__date__ = "August 26th 2022"

from global_variables import *
from grad2vec import Grad2Vec

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import k_means
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import AgglomerativeClustering

class Clustering:

    def __init__(self,g2v_instance,alg,true_labels=None,ratio=None,weight_step=0):
        '''Clustering - A class that when provided with an initialized Grad2Vec instance can perform clustering on its embeddings to find clusters of similar records.

        Inputs:

        g2v_instance : Grad2Vec
        Initialized Grad2Vec instance that has embeddings generated and stored internally. See grad2vec.py for more information on the Grad2Vec class

        alg : Scitkit learn clustering algorithm or pipeline
        Allows for your choice of clustering algorithm or handcrafted clustering pipeline to be passed in for the clustering task. So long as the clustering algorithm has a `fit_predict()` method, it should be compatible with this class. Defaults to K-Means with standardization, see https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html and https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html for more information.


        true_labels : int[]
        If the format of your labels differs from the convention followed in preprocessing.py, this allows for the values of the known duplicate labels to be passed in. Only worry about this if you are not using the MAEPy pipeline.

        ratio : float
        Ratio of duplicates to non-duplicates used for determining which labels to focus on in evaluating clustering accuracy. Only worry about this if you are not using the MAEPy pipeline.


        weight_step : 0
        In short, this indicates at what point the weights from the classifier are added into the embeddings, and does not need to be considered unless you are providing in your own scikit learn pipeline. If you have a pipeline that you are passing in, it is probably the case that you want certain preprocessing steps to take place before skewing the embeddings with the weights. Providing the `weight_step` parameter allows you to index into the scikit pipeline at whichever point you would like and have the embeddings skewed there. See `fit_predict()` source-code to witness the use of `weight_step` in action


        Resulting Attributes:

        g2v : Grad2Vec() Instance

        alg : Scitkit learn pipeline w/ clustering estimator

        true_labels : int[]

        weight_step : int


        Notes:

        For internal consistancy, any provided clustering algorithm will be automatically placed into a scikit learn pipeline, meaning that modifying the `alg` parameter directly might result in the program crashing. It is recommended in that case to just to initialize a seperate Cluster instance. 

        To learn more about how scikit learn pipelines work, visit https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
        and to see more options of supported clustering algorithms, visit https://scikit-learn.org/stable/modules/clustering.html 

        '''

        self.g2v = g2v_instance 
        self.alg = alg
        # Place provided algorithm into Scikit pipeline for consistency
        if str(type(alg)) != "<class 'sklearn.pipeline.Pipeline'>":

            self.alg = make_pipeline(alg)
            weight_step = 0

        self.true_labels = true_labels
        self.weight_step = weight_step

        # This labels the known duplicates automatically which is used for evaluating the accuracy of the clustering
        if self.true_labels == None:
            assert ratio != None
            self.true_labels = self._get_default_labels(ratio)

    def fit_predict(self,n_clusters=None,weight_exp=8,set_attrs=True):
        '''fit_predict(self,n_clusters=2,weight_exp=8,set_attrs=True)
        Performs clustering on the Grad2Vec embeddings with respect to the provided `alg`in the initialization step. This results in an array of labels `predicted_labels` which denotes the found clusters and can be used in future steps.

        Inputs:

        n_clusters : int
        Number of clusters to be used. Leaving this unset just defaults to however many was set in the initialized `alg` model. Note that certain clustering models do not allow manual control over the number of clusters, it is best to verify that your provided model in `alg` has a `n_clusters` parameter if

        weight_exp : int or float
        When applying weights to the embeddings in Grad2Vec, it has been discovered experimentially that raising the weights (element-wise) to a higher degree than 1 improves the clustering accuracy. Intuitively, this just further skews the weights with respect to their importance. Results tend to do well when raised to the 8th power, but this parameter provides flexibility in your choice of the degree.

        set_attrs : Bool
        If True, sets the resulting clustering labels to `predicted_labels`. (This allows for other methods like `plot_elbow` to work without affecting the labels)

        Returns:

        predicted_labels : int[]
        List of labels cooresponding to which cluster each record was chosen to be a member of.

        '''
        weights = self._get_weights()
        embeddings = self.g2v.embeddings 

        #Update the number of clusters if requested
        if type(n_clusters) == int:
            
            #Note: This updates the parameters of the model itself, which remains changed even after this function has finished
            try:
                self.alg[-1].n_clusters = n_clusters

            except:
                print('Error: Provided clustering algorithm has non-supported interface for updating cluster amount. Either manually update your model yourself, verify that the provided clustering approach allows control over the number of clusters, or set `n_clusters` as None and try again.')
                return
        

        #NOTE: Potential way to reduce memory usage would be to design a class that supports a `fit_transform()` method that scales an input with respect to the weights (rather than splitting the pipeline as done below). That way, this weight class could be passed into the sklearn pipeline which means the sklearn could be run without needing to stop in the middle to add the weights.

        #Perform all preprocessing before applying weights
        if self.weight_step != 0:
           embeddings = self.alg[0:self.weight_step].fit_transform(embeddings) 
          
        # Apply the weights, perform all subsequent steps and then cluster
        predicted_labels = self.alg[self.weight_step:].fit_predict( embeddings * np.power(weights,weight_exp))

        if set_attrs:
            self.predicted_labels = predicted_labels
            self._weight_exp = weight_exp


        return predicted_labels


    def set_clusters(self,colname=CLUSTER_COL):
        '''set_clusters(self,colname=CLUSTER_COL)
        Given `fit_predict()` has been called, sets the predicted labels into the `g2v.records` DataFrame. This is a convenience function which is used primarly in the MAEPy pipeline.

        Input:

        colname : str
        The name of the column that the labels will be set to in the records DataFrame. Defaults to the value set for `CLUSTER_COL` which can be changed in global_variables.py

        
        '''
        assert hasattr(self,'predicted_labels')
        self.g2v.records[colname] = self.predicted_labels


    def score(self,predicted_labels=None,true_labels=None,as_percent=True):
        '''score(self,predicted_labels=None,true_labels=None,as_percent=True) 
        For a given set of true duplicate labels, returns accuracy of predicted cluster labels based on how many of the true duplicates are placed in any of the predicted clusters.

        Inputs:

        predicted_labels : int[]
        NumPy array of labels cooresponding to predictions. Defaults to what is set in `predicted_labels`.

        true_labels : int[]
        NumPy array of labels cooresponding to actual known labels. Defaults to what is set in `true_labels`.

        as_percent : Bool
        If true, returns the score as a percent, rather than a tally of correctly placed duplicates.
        
        
        Returns:

        tally : int or float
        Cooresponds to either the tally or percentage of correctly placed duplicates (w.r.t `as_percent`).

        '''

        if type(true_labels) == type(None):
            true_labels = self.true_labels
        if type(predicted_labels) == type(None):
            predicted_labels = self.predicted_labels
            
        labels = np.unique(true_labels)
        tally = 0
        for lab in labels:
            #Here, the 0 label indicates non-duplicates (we ignore them)
            if lab != 0:
                pred = predicted_labels[true_labels == lab]
                if pred.min() == pred.max():
                    # Sucess, we have a correct prediction!
                    tally += 1

        if as_percent:
            return tally/(len(labels)-1)

        else:
            return tally


    def plot_elbow(self,min_components=2,max_components=50,step=1,weight_exp = 8,true_labels=None,as_percent=True):
        '''plot_elbow(self,min_components=2,max_components=50,step=1,weight_exp = 1,true_labels=None,as_percent=True)
        Plots a graph of the accuracy of the clustering with respect to the number of cluster components. Helps in gaining an idea of how many clusters to select.

        Inputs:

        min_components : int
        Minimum number of components to use

        max_components : int
        Maximum number of components to use

        step : int
        How much should each step from `min_components` to `max_components` be

        weight_exp : int
        What power to raise the weights by used to skew the embeddings.

        true_labels : int[]
        List of the true labels to compare against for each operation plotted

        as_percent : Bool
        If true, plots the score as a percent, rather than a tally of correctly placed duplicates.

        Returns

        Plot of the results, x-axis cooresponding to the number of components and the y-axis cooresponding to the accuracy. See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html for details on the plotting technique.

        '''

        if true_labels == None:
            true_labels = self.true_labels
 
        component_sizes = np.arange(min_components,max_components + 1,step)
        component_scores = np.zeros(shape= component_sizes.shape)
        for i, val in enumerate(component_sizes):

            predicted_labels = self.fit_predict(n_clusters=val,weight_exp=weight_exp,set_attrs=False)
            
            component_scores[i] = self.score(predicted_labels,true_labels,as_percent)

        plt.scatter(component_sizes,component_scores)
        plt.xlabel('Number of cluster components')
        plt.ylabel('Accuracy')

    def get_false_negatives(self):
        '''get_false_negatives(self) 
        Given that clustering has taken place (i.e `fit_predict()` called), returns all pairs that ended up being placed in seperate clusters.

        Returns:

        false_neg : Pandas DataFrame
        Table of duplicates, each adjacent to each-other similar to how done in `g2v.records`

        '''
        false_neg = []
        predicted_labels = self.predicted_labels 
        true_labels = self.true_labels 
        labels = np.unique(true_labels)
        for lab in labels:
            #Here, the 0 label indicates non-duplicates (we ignore them)
            if lab != 0:
                pred = predicted_labels[true_labels == lab]
                
                if pred.min() != pred.max():
                    # A pair has been incorrectly seperated; thus false negative

                    cur_cluster = self.g2v.records.iloc[true_labels == lab]
                    false_neg.append([cur_cluster])

        return false_neg

    def find_cluster(self,record):
        '''find_cluster(self,record)
        For an inputted record, finds and returns the cluster in which it was placed in. This allows for the results of a `fit_predict()` to be examined without needing to call `set_clusters()`, allowing for easier analysis.

        Input:

        record : Pandas Series or Panadas DataFrame
        Individual record from `g2v.records` to search for

        Output:

        cluster : Pandas DataFrame
        A table containing all of the records that were placed in the same cluster (includs the inputted record)


        '''
        assert hasattr(self,'predicted_labels')
        
        #Grab the index of the record w.r.t `g2v.records` df
        if type(record) == type(self.g2v.records):
            index = record.index[0]
        else:
            index = record.name

        label = self.predicted_labels[index]

        return self.g2v.records.iloc[ self.predicted_labels == label]


    def _get_default_labels(self,ratio):
        '''_get_default_labels(self,ratio)
        Helper function which assumes that duplicates all are placed in pairs and are listed at the top of `records` with respect to the size of ratio. With that assumption, the function creates labels that identify duplicates (as pairs with positive integer labels) and non-duplicates (labels all 0).

        Input:

        ratio : float
        The ratio of duplicates in `g2v.records` assumed to be duplicates (and in the top of the dataset).

        Returns:

        labels : int[]
        NumPy array of true labels
        
        '''
        labels = np.zeros(shape=(self.g2v.n_records))
        dup_length = int((self.g2v.n_records * ratio)/2)
        labels[0:2*dup_length:2] = np.arange(1,dup_length+1)
        labels[1:2*dup_length:2] = np.arange(1,dup_length+1)
        return labels

    def _get_weights(self):
        '''_get_weights(self)
        Returns the weights used for skewing the embeddings in the right shape so that the multiplication of the embeddings by the weights can be "broadcasted" (that is, the operation will be vectorized). This design choice allows for different classifiers to be used without having to compute a new set of embeddings each time. See https://numpy.org/doc/stable/user/basics.broadcasting.html for more information on broadcasting.

        Note that if weights are not set, this function just returns an array of ones, which has no effect on the values of the embeddings.

        Output:

        weights : float[]
        NumPy array of weights, of length (num_columns x num_splits)
        '''
        if self.g2v._skew_embeddings:
            #This just "stretches" the weights to match the width of the embeddings
            return np.tile(self.g2v.weights,(1,self.g2v.splits))

        else:
            return np.ones(len(self.g2v.cols)*self.g2v.splits)

# ---------- Miscellaneous helper functions ----------

def num_clusters(N,scale=np.log2):
    '''Small tool used for calculating how many clusters needed for dedupe algorithm to perform O(N scale(N)) comparisons (where N is the size of `g2v.records`. This is assuming that the cluster sizes would be identical (not necessarily true in practice).'''
    k = N / scale(N)
    return k
