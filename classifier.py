#! /usr/bin/env python3

__author__ = "Michael Ripa"
__email__ = "michaelripa@cmail.carleton.ca"
__version__ = "1.0.0"
__date__ = "August 26th 2022"


from global_variables import *
from similarity import Similarity
from preprocessing import load_df, find_pairs, check_transposition, create_training_labels

import pandas as pd
import numpy as np
from copy import deepcopy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class Classifier:

    def __init__(self,similarity,n=None,classifier=LogisticRegression(),train_ratio = 0.5,standardize=True,df=GRADUATES_PATH):
        
        '''Classifier - A class for initializing and training a classifier that is coupled to a Similarity instance for transforming non-numerical data into something that can be trained with a classifier model. 

        Inputs:

        similarity : Similarity
        Similarity instance used for generating numerical distance vectors used for the classifier.

        n : int
        Number of duplicates to train on. Defaults to training on all of the duplicates (ideal for use in production)


        classifier : Scikit learn classifier instance 
        Linear classifier to be used by the model with hyperparameters of one's choice. Can incorporate a scikit learn pipeline granted the final estimator in the pipeline is a linear classifier. See https://scikit-learn.org/stable/modules/linear_model.html for examples of linear models that can be used and https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html for more details on scikit learn pipelines.

        train_ratio : float
        Ratio of duplicates to non-duplicates to train on. 

        standardize : bool
        If True, performs standardization on the data before passing it into the classifier for training. Tends to improve model accuracy, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html for more details.

        df : str or Pandas DataFrame
        Path to dataset or Pandas df used for training the classifier.


        Resulting Attributes:

        df : Pandas DataFrame
        DataFrame to be sampled from for training

        n : int
        Number of duplicates (to be) used in training

        clf : Sklearn Pipeline
        Pipeline setup with linear classifier

        sim : Similarity() 
        Similarity instance used for computing distances

        train_ratio : float
        Ratio of duplicates to non-duplicates


        '''

        if type(df) == str:
            try:
                self.df = load_df(df,default_preprocessing=True)
            except:
                raise ValueError('Could not read ' + df + ' as a csv, please verify that the path is correct.')
        else:
            self.df = df

        self.clf = classifier
        if type(classifier) != sklearn.pipeline.Pipeline:
            # Place the provided classifier into Scikit pipeline for consistency
            if standardize:
                self.clf = make_pipeline(StandardScaler(),classifier)
            else:
                self.clf = make_pipeline(classifier)


        self.n = n
        self.clf = classifier
        self.train_ratio = train_ratio
        self.sim = similarity
        
    def init_classifier(self,training=False,test_size=0.4):
        '''init_classifier(self,duplicates_df=DUPLICATE_GRADUATES_PATH,training=False,standardize=False)
        Trains classifier model on pairs of duplicates and non-duplicates with repect to the hyperparameters passed in at initialization.

        Inputs:

        training : bool
        If True, Creates a train test split of ratio 0.4, storing attributes `X_train`, `X_test`, `y_train` and `y_test` and training the classifier on `X_train`, `y_train`. If False, all of the data is used to train the model but none of it is stored.

        test_size : float
        Given `training` is True, this decides the ratio of the test set size to the train set size (i.e. ratio of `X_test` to `X_train`). 

        Resulting attributes:

        n : int

        n_true : int
        Number of duplicates

        n_false : int
        Number of non-duplicates

        '''
        pairs = find_pairs(df=self.df,similarity=self.sim,n=self.n,duplicates=True,strict_pair=True,move_to_top=False,return_ratio=False)

        #Compute number of non-duplicates with respect to train_ratio. Note that this computation works even if n = None.
        n_true = len(pairs)
        n_false = int((n_true - self.train_ratio*n_true)/self.train_ratio)

        non_pairs = find_pairs(df=self.df,similarity=self.sim,n=n_false,duplicates=False,strict_pair=True,move_to_top=False,return_ratio=False)

       
        #Compute distance vectors
        X_true = self.sim.compute_similarities(pairs)
        X_false = self.sim.compute_similarities(non_pairs)

        X,y = create_training_labels(X_true,X_false)

        
        if training: 
            #Generates training and test sets and stores internally
            X, X_test, y, y_test = train_test_split(X,y,test_size=test_size)
            self.X_train = X
            self.y_train = y
            self.X_test = X_test
            self.y_test = y_test


        self.clf.fit(X,y)

        self.n = n_true + n_false
        self.n_true = n_true
        self.n_false = n_false

