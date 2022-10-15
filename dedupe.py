#! /usr/bin/env python3

__author__ = "Michael Ripa"
__email__ = "michaelripa@cmail.carleton.ca"
__version__ = "1.0.0"
__date__ = "August 26th 2022"

from global_variables import *
from classifier import Classifier
from similarity import Similarity
from preprocessing import create_training_labels, compute_all_pairs

import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, recall_score, f1_score
from scipy.special import comb

class Dedupe:

    def __init__(self,cluster_df,clf,similarity,cluster_col=CLUSTER_COL,duplicate_col=DUPLICATE_COL):
        '''Dedupe : A class used for detecting and identifying pairs of duplicates in a dataset. 

        Inputs:

        cluster_df : Pandas DataFrame
        Dataset of records containing a column of cluster labels, indicated by `cluster_col`. Only pairs with the same `cluster_col` labels get compared, meaning that you need to set the values of `cluster_col` to a constant value if you want to compare over the whole DataFrame.

        clf : Scikit learn classifier 
        Classifier to be trained and used in predicting whether pairs of entries are duplicates or not. Note that in this case, the classifier cannot be a Classifier class, and needs to have `fit(X,y)` and `predict(X)` methods. A work-around for using more advanced classifiers (e.g. Tensorflow feedforward neural networks) is creating a wrapper class that has custom defined `fit(X,y)` and `predict(X)` methods. Passed in model can also be a scikit learn pipeline, given the final estimator supports the above methods.

        similarity : Similarity
        Similarity instance which contains the columns, distance functions and match column used for classification. Note that in context of the MAEPy pipeline, this similarity instance need not be identical to the one used in generating the Grad2Vec embeddings.

        cluster_col : str
        Name of column containing the cluster labels. Defaults to the value of `CLUSTER_COL` found in global_variables.py

        duplicate_col : str
        Name to be given to the column containing the duplicate labels. Defaults to the value of `DUPLICATE_COL` found in global_variables.py

        
        Resulting Attributes:

        clf : Classifier of choice
    
        cluster_df : Pandas DataFrame

        similarity : Similarity() instance

        cluster_col : str

        duplicate_col : str

        '''

        cluster_df
        self.clf= clf # Note: Cannot be a Classifier() instance
        self.cluster_df = cluster_df
        self.similarity = similarity
        self.cluster_col = cluster_col
        self.duplicate_col = duplicate_col
        self._distance_cache_set = False


    def train_classifier(self,n=None,ratio=0.05,training=False,test_size=0.4):
        '''train_classifier(self,ratio=0.05,training=False,test_size=0.4)
        Trains provided classifier instance in `clf` on inner-cluster duplicates/non-duplicates, i.e. does not train on a pair if they come from different clusters.

        Inputs:

        n : int
        Number of duplicates placed in same cluster to be trained on (w.r.t `similarity.match_col`). Defaults to training on all possible duplicates

        ratio : float
        Ratio of duplicates to non-duplicates found in same cluster to be trained on (w.r.t `similarity.match_col`).

        training : bool
        If True, sets attributes `X_train`, `X_test`, `y_train` and `y_test` to instance. This provides a way of testing the accuracy of the classifier directly.

        test_size : float
        Given that `training` is set to True, controls the ratio of test data to train data. Cooresponds to `test_size` ratio of sklearn API, see https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html for details.

        '''
        # Avoid dividing by zero
        assert ratio != 0
        
        #TODO: This might not be optimal for the case where one is experimenting with classifiers. Requires computing all distances, rather than just ones to be used for training
        if self._distance_cache_set == False:
            self._compute_distances()

        dups = self._distance_cache[self._labels_cache == 1]
        non_dups = self._distance_cache[self._labels_cache == 0]
        

        n_true = n
        if type(n) == type(None):
            n_true = len(dups)

        n_false = int((n_true/ratio)- n_true)

        #Now we determine if there are enough duplicates/non-duplicates to train on
        try:
            assert n_true <= len(dups)
            assert n_false <= len(non_dups)

        #In the event that there are too few duplicates or non-duplicates, prompt user whether to proceed or exit.
        except:
            print('Warning: Requested training set size too large, can only train on ' + str(len(dups)) + ' duplicates and ' + str(len(non_dups)) + ' non-duplicates ( ratio = ' + str(round(len(dups)/len(non_dups),5)) + ')')

        X,y = create_training_labels(dups,non_dups)        
        if training: 
            #Generates training and test sets and stores internally
            X, X_test, y, y_test = train_test_split(X,y,test_size=test_size)
            self.X_train = X
            self.y_train = y
            self.X_test = X_test
            self.y_test = y_test

 
        self.clf.fit(X,y)
        return



    def get_duplicates(self,n=None,training=False,test_size=0.4,ratio=0.5):
        '''get_duplicates(self,training=False,test_size=0.4,ratio=0.5)
        Goes through the clusters labeled in `cluster_df`, computes pairwise distances, and then trains and runs a classifier on said distances in order to predict pairs of duplicates. The resulting predictions are then added to a new column of `cluster_df` called `duplicate_col`.

        Inputs:

        n : int
        Number of duplicates to train on. Defaults to training on all possible duplicates.

        training : Bool
        If true, then the training set will be split (w.r.t `test_size` ratio) and `X_train`,`X_test`,`y_train` and `y_test` attributes will be added

        test_size : float
        If `training` set to True, this will divide the training data into a train set consisting of (1-test_size)% of the data and a test set with test_size % of the data.

        ratio : float
        Ratio of duplicates to non-duplicates to be trained on. If not enough data exists to accomadate the provided ratio, a warning is printed, but the operation still carries on.

        Resulting Attributes:

        predicted_labels : int[]
        Array containing the predicted labels of duplicates. Format is a string of the form 'a b c '.

        '''
     

        if self._distance_cache_set == False:
            self._compute_distances()

        self.train_classifier(n=n,training=training,test_size=test_size,ratio=ratio)

        y_hat = self.clf.predict(self._distance_cache)

        self.cluster_df[self.duplicate_col] = np.empty(len(self.cluster_df),str)
        dup_index = (y_hat == 1)
        self.predicted_labels = y_hat

        # Populate the `duplicate_col` column with duplicate labels
        for (left,right,i) in zip(self._left_index[dup_index],self._right_index[dup_index],np.arange(1,np.sum(dup_index) + 1)):
            self.cluster_df[self.duplicate_col].iloc[left] += str(i) + ' '
            self.cluster_df[self.duplicate_col].iloc[right] += str(i) + ' '

        
    def _compute_distances(self):
        '''_compute_distances(self) 
        A very important helper function in the deduplication process, responsible for a number of different tasks. The most important procedure is in building two DataFrames of identical shape in a format supported by `similarity.compute_similarities()` and then returning their elementwise distances. These DataFrames will represent all pairs within the clusters needing to be compared against eachother.

        Resulting Attributes:

        A number of pieces of "hidden" data, such as all of the distance computations and their labels, used elsewhere in the class.

        '''

        num_comparisons = self.count_comparisons(compare_cartesian=False)
        label_df = self.cluster_df.reset_index()[self.cluster_col]

        #To perform all comparisons, we first create 2 DataFrames of equal length that we use to pass into `similarity.compute_similarities()`        

        index_1 = np.zeros(num_comparisons)
        index_2 = np.zeros(num_comparisons)
        cur_position = 0

        # This whole procedure is just iterating all of the clusters and adding index pairs that we want to compare into index_1 and index_2 (to be used in creating DataFrames)
        for label in label_df.unique():
            
            #Grab index values of current cluster records
            cluster_index = label_df.index[label_df == label].to_numpy()
            cluster_len = len(cluster_index)
            
            if cluster_len > 1:
                # This step would take [a,b,c,d] and obtain [a,a,a,b,b,c] , [b,c,d,c,d,d]
                n_comparisons = int(comb(cluster_len,2))
                new_position = cur_position + n_comparisons

                #See preprocessing.py for implementation
                left_index_values , right_index_values = compute_all_pairs(cluster_index,combine=False)
                index_1[cur_position: new_position] += left_index_values
                index_2[cur_position: new_position] += right_index_values

                # Done, increment our position and breathe a sigh of relief!
                cur_position += n_comparisons
                

        # All this work above (asides from classifying) was to get our pairs to compare in two equal length DataFrames, now can take advantage of the wicked fast vectorization!

        df_1 = self.cluster_df.iloc[index_1,:].reset_index()
        df_2 = self.cluster_df.iloc[index_2,:].reset_index()

        #Now we want our labels for the classifier, which is what y_known encapsulates
        y_known = np.zeros(len(index_1))
        match_col = self.similarity.match_col

        #Here 1 means duplicate 
        y_known[df_1[match_col]== df_2[match_col]] = 1
        #Here -1 means unknown (i.e. not enough information w.r.t match_col
        y_known[df_1[match_col].isna() | df_2[match_col].isna()] = -1

        #The most lengthy of tasks throughout the entire pipeline: Computing all cluster distances
        distances = self.similarity.compute_similarities(df_1,df_2)
        
        #Done, cache information away for later useage
        self._distance_cache = distances
        self._labels_cache = y_known
        self._left_index = index_1.astype(int)
        self._right_index = index_2.astype(int)
        self._distance_cache_set = True



    #TODO: This should only compute on the test set in practice
    def score(self,precision=True,recall=True,sensitivity=True,accuracy=True,n_decimals=5):
        '''score(self,precision=True,recall=True,sensitivity=True,accuracy=True,n_decimals=5)
        Evaluates the classifier on all compared pairs that have reliable `similarity.match_col` values in place (i.e our "known" duplicates and non-duplicates). Allows in choice between one or more of the following performance metrics: precision, recall, sensitivity (or true negative rate) and accuracy.

        Inputs:

        precision : Bool
        If True, returns the precision score
        
        recall : Bool
        If True, returns the recall score

        sensitivity : Bool
        If True, returns the sensitivity score

        accuracy : Bool
        If True, returns the accuracy score

        n_decimals : int
        Number of decimals each score gets rounded to
        '''
        
        assert hasattr(self,'predicted_labels')

        filter_mask = (self._labels_cache != -1)
        
        #Filter out pairs which don't have reliable information
        known_true = (self._labels_cache == 1)[filter_mask]
        predicted_true = (self.predicted_labels == 1)[filter_mask]
        known_false = (self._labels_cache == 0)[filter_mask]
        predicted_false = (self.predicted_labels == 0)[filter_mask]

        tp = np.sum(known_true & predicted_true)
        fp = np.sum(~known_true & predicted_true)
        tn = np.sum(~known_true & ~predicted_true)
        fn = np.sum(known_true & ~predicted_true)
        
        prec =  tp/(tp + fp)
        rec = tp/(tp + fn)
        sens = tn/(tn + fp)
        acc= (tp + tn)/(tp + tn + fp + fn)

        if precision:
            print('Precision: ' + str(round(prec,n_decimals)))
        if recall:
            print('Recall: ' + str(round(rec,n_decimals)))
        if sensitivity:
            print('Sensitivity: ' + str(round(sens,n_decimals)))
        if accuracy:
            print('Accuracy: ' + str(round(acc,n_decimals)))
                


    def count_comparisons(self,compare_cartesian=False):
        '''count_comparisons(self,compare_cartesian=False)
        Counts the total number of pairs that needs to be compared post-clustering with respect to `cluster_df` and the labels selected in `cluster_col`. Can be used for asserting cluster size effectiveness by comparing the number of pairs formed by entries restricted to their own clusters against the number of pairs formed across the entire dataset (i.e cartesian product). Also used internally within different class functions.

        Inputs:

        compare_cartesian : Bool
        If True, prints out an informative message containing both how many comparisons would have taken place with and without clustering and displays the fraction of the two counts.

        Returns:

        num_comparisons : int
        Count of how many comparisons will be needed to be computed. Note that this exludes self comparisons (i.e. a record with itself) and counting permutations (i.e considering a distinction in the order a pair is compared).

        '''
        label_df = self.cluster_df[self.cluster_col]
        num_comparisons = 0
        total_records = len(label_df)

        # Iterate over each label and count the number of pairs
        for label in label_df.unique():
            cluster = label_df[label_df == label]
            cluster_freq = len(cluster)
            #If cluster contains one record, no comparisons need to be done
            if cluster_freq > 1:
                num_comparisons += comb(cluster_freq,2)

        if compare_cartesian:
            num_cartesian = int(comb(total_records,2))
            ratio = num_comparisons/num_cartesian
            print('Only ' + str(int(num_comparisons)) + ' cluster comparisons needed instead of ' + str(num_cartesian) + ' comparisons  (' + str(round(ratio,5)) + ' % total comparisons)')
            return
    
        return int(num_comparisons)

