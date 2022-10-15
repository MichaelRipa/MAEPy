#! /usr/bin/env python3

__author__ = "Michael Ripa"
__email__ = "michaelripa@cmail.carleton.ca"
__version__ = "1.0.0"
__date__ = "August 26th 2022"

from global_variables import *
from preprocessing import load_df, find_pairs
from similarity import Similarity
from classifier import Classifier
from clustering import Clustering
from grad2vec import Grad2Vec 
from dedupe import Dedupe

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#BisectingKMeans chosen as default clustering algorithm, but do check out KMeans and MiniBatchKMeans as alternatives, as they have their tradeoffs.
from sklearn.cluster import FeatureAgglomeration ,BisectingKMeans

# Suppresses printing of unhelpful warnings related to certain operations performed on Pandas DataFrame in the pipeline. Does not appear to have any effect on functionality, but if you are wanting further information, see https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy 
pd.set_option('mode.chained_assignment',None)


class MAEPy:


    def __init__(self,cols=default_cols,graduate_path=GRADUATES_PATH,dup_graduate_path=DUPLICATE_GRADUATES_PATH,nrows=None):
        '''MAEPy - A pipeline for performing deduplication on PSRS Graduates data (or any dataset that you want to detect duplicates in). 

        Inputs:

        cols : str[] 
        List of columns to be used throughout the pipeline. Defaults to what default_cols is set to in global_variables.py

        graduate_path : str 
        Path to where PSRS graduates table is stored locally. Defaults to what GRADUATES_PATH is set to in global_variables.py

        dup_graduate_path : str 
        Path to where PSRS graduates table of duplicates is stored locally. Defaults to what DUPLICATE_GRADUATES_PATH is set to in global_variables.py

        nrows : int
        Number of rows to read from the dataset. Defaults to the whole dataset.


        Works in harmony with the following components:

        sim : Similarity() instance
        Used for comparing pairs of records and computing their distances. Also stores meta-information important for functionality within the proceeding components. Generated from `init_similarity()`.

        classifier : Classifier() instance
            Used for training and evaluating a linear classifier. Generated from `init_classifier()` function call.

        g2v : Grad2Vec() instance 
            Used for creating embeddings (vector representations). Generated from `init_embeddings()` function call.

        cluster : Clustering() instance 
            Used for clustering on the generated embeddings in finding groups of similar entries. Generated from `init_clustering()` function call.

        dedupe : Dedupe() instance 
            Used for sifting through the clusters and determining pairs and groups of duplicates. Generated from `init_dedupe()` function call.


        Resulting Attributes (from initialization)

        cols : str[]
        df : Pandas DataFrame

    '''
        self.cols = cols
        self.df = load_df(path=graduate_path,default_preprocessing=True,nrows=nrows)
 

    def init_similarity(self,cols=default_cols,match_col=match_col,comp_functions=default_comp_functions):
        '''init_similarity(self,cols=default_cols,match_col=match_col,comp_functions=default_comp_functions)
        Constructs a Similarity instance with respect to the provided parameters. This should be the first function called within the pipeline (after initialization).

        Inputs:

        cols : str[]
        List of columns that will be compared against. Note that this defaults to the value of `default_cols` in global_variables.py

        match_col : str
        Column used for obtaining training examples of both duplicate and non-duplicate pairs. Defaults to the value of `match_col` set in global_variables.py

        comp_functions : dict{}
        Determines how each column is compared when computing distances. See global_variables.py and similarity.py for information on how to modify this parameter (defaults to the value of `comp_functions` set in global_variables.py).


        Resulting Attribute:

        sim : Similarity()
        The initalized Similarity instance.

        '''
        self.sim = Similarity(cols=cols,match_col=match_col,comp_functions=comp_functions)

    def init_classifier(self,n=None,similarity=None,classifier=None,train_ratio=0.5,standardize=True,training=False,train_test_split_ratio=0.4):
        '''init_classifier(self,n=None,similarity=None,classifier=None,train_ratio=0.5,standardize=True,training=False)
        Constructs a Classifier instance and internally trains a linear classifier on all the duplicates found in the seperate duplicates df (dup_df) set up on initialization.

        Inputs:

        n : int 
        Number of duplicate pairs to train classifier on. Defaults to all possible pairs found within the dataset.

        similarity : Similarity()
        Similarity instance used internally for creating the numerical distance vectors used in the training process. Defaults to the pre-initialized `sim` instance.

        classifier : Initialized Scikit learn linear classifier (or pipeline)
        Allows for different classifiers or handcrafted classifier pipelines to be passed in, allowing control over hyperparameter and model choice. Note that in order for this function to work, the classifier (or pipeline) MUST have the same interface as the Scikit Learn library (in particular, a fit(X,y) method). In order to use this classifier to assist with the embedding step, the classifier MUST be linear (having a coef_ attribute after training). See scikit-learn.org/stable/modules/linear_model.html for more classifier options. 
        Defaults to Logistic Regression with Standardization, see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html and https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html for more information.

        train_ratio : float
        Ratio of duplicates to non-duplicates used in training.

        standardize : bool
        If True, will standardize the data used for training the classifier. Standardization transforms the datset column-wise to take on a sample mean of 0 and a sample std of 1 (making each column tend to the same distribution for large n)
        
        training : bool
        If True, sets attributes X_train, X_test, y_train, y_test to classifier. This provides a way of testing the accuracy of the classifier directly.

        train_test_split_ratio : float
        Given that `training` is set to True, controls the ratio of test data to train data. Cooresponds to `test_size` ratio of both Classifier and sklearn API. 

        '''
        if type(similarity) == type(None):
            if hasattr(self,'sim') == False:
                raise ValueError('No pre-existing Similarity instance existiher make call to init_classifier() (recommended) or set skew_embeddings=False')
            similarity = self.sim
        if type(classifier) == type(None):
            if standardize:
                classifier = make_pipeline(StandardScaler(),LogisticRegression())
                
            else:
                classifier = make_pipeline(LogisticRegression())
        
        self.classifier = Classifier(n=n,similarity=similarity,classifier=classifier,train_ratio=train_ratio,standardize=standardize,df=self.df)
        self.classifier.init_classifier(training=training,test_size=train_test_split_ratio)
        self.standardized=standardize

    def init_embeddings(self,n=None,ratio =0.1,k=10,splits=10,similarity=None,skew_embeddings=True,basis=None): 
        '''init_embeddings(self,n=None,ratio =0.1,k=10,splits=10,similarity=None,skew_embeddings=True,basis=None)
        Initializes a Grad2Vec instance and uses it to generate embeddings for the entire dataset provided in the initialization step. 

        Inputs:

        n : int
        Number of pairs of duplicates to keep track of. This is used to gain insight into the accuracy of the model, and only these pairs will be tracked in subsequent steps of the pipeline. Defaults to all possible pairs, in which case the `ratio` parameter is ignored.

        ratio : float
        Given that a value for n is provided, this determines the ratio of pairs of duplicates to pairs of other records (i.e. for 1000 duplicate pairs and a ratio of 0.05, then 40000 records would be embedded). If n is not provided, this value is ignored.

        k : int
        Number of records sampled and used in generating embeddings (i.e. "basis dimension"). 
        splits : int
        When creating the embeddings, how many "splits" of the k records compared against are made. Controls the width of the embedding directly.

        similarity : Similarity()
        Similarity instance used internally for creating the numerical distance vectors used in the training process. Defaults to the pre-initialized `sim` instance.

        skew_embeddings : bool
        If True, the weights of the linear classifier in `classifier` are used to skew the embeddings.

        basis : Pandas DataFrame
        Records used for comparison when creating the embeddings. Each record gets its distance computed with each basis element which is used to generate the embeddings (w.r.t `splits`). Defaults to a random selection based on the choice of `k` value. Note that `k` needs to match the length of `basis`.

        Resulting Attributes:

        ratio : float
        The resulting ratio of duplicate pairs to non-duplicates, used in future steps of the pipeline

        g2v : Grad2Vec() 
        Initialized Grad2Vec instance
        
        '''
        if skew_embeddings and hasattr(self,'classifier') == False:
            raise ValueError('Classifier has not been initialized and trained, either make call to init_classifier() (recommended) or set skew_embeddings=False')

        if type(similarity) == type(None):
            similarity = self.sim

        #We return a ratio in case we are working on the entire dataset and do not know the ratio in advance. 
        # The additional_cols parameter ensures that the dataset maintains a unique identifier for each record.
        pairs_df , obs_ratio = find_pairs(df=self.df,similarity=similarity,n=n,duplicates=True,strict_pair=True,move_to_top=True,return_ratio=True,subset_cols=True,additional_cols=[unique_identifier])

        
        if n != None:
            #Here, we actually take the value of `ratio` into account
            n_false = int((n/ratio)- n)
            pairs_df = pairs_df.iloc[0: 2*n + 2*n_false]
        else:
            ratio = obs_ratio
        
        self.ratio = ratio

        #Initalize Grad2Vec instance and use it to embed our data.
        self.g2v = Grad2Vec(similarity=similarity,records=pairs_df)
        self.g2v.generate_embeddings(k=k,splits=splits,default_basis=basis,apply_changes=True,apply_mean=True)
        
        #This uses the weights of the classifier to skew the embeddings so that big differences in important columns is amplified in the embedding and vice versa
        if skew_embeddings:
            weights = self.classifier.clf[-1].coef_
            self.g2v.set_classifier_weights(weights)


    def init_clustering(self,alg=None,n_clusters=400,weight_exp=8,true_labels=None,print_score=False,weight_step=0,colname=CLUSTER_COL):
        '''init_clustering(self,alg=None,n_clusters=400,weight_exp=8,true_labels=None,print_score=False,weight_step=0)
        Initializes a Clustering instance with the embeddings found in Grad2Vec instance and performs clustering with respect to the provided hyperparameters. On completion, this also adds a new column to `g2v.records` containing the predicted labels, which is used by Dedupe in `init_dedupe()`. Note that further experimentation and tweaking can then take place through the `cluster` attribute.

        Inputs:

        alg : Scitkit learn clustering algorithm or pipeline
        Allows for your choice of clustering algorithm or handcrafted clustering pipeline to be passed in for the clustering task. So long as the clustering algorithm has a `fit_predict()` method, it should be compatible with this class. Defaults to Bisecting K-Means with standardization, see https://scikit-learn.org/stable/modules/generated/sklearn.cluster.BisectingKMeans.html and https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html for more information.


        n_clusters : int
        Number of clusters to be discovered by the clustering algorithm. Note that the resulting `cluster` attribute has functionality to help explore the accuracies in different clustering amounts that can be used (e.g plot_elbow method)

        weight_exp : int
        If embeddings in `g2v` have the `weight_exp` attributes set, then applies the weights to the embeddings before classifying, otherwise has no effect.

        true_labels : int[] 
        List of integers corresponding to the true labels of the embeddings in self.g2v. Ignoring this step is fine so long as self.ratio is set (which is done automatically from init_embeddings call) and that pairs of duplicates are kept next to each other at the top of the DataFrame (done automatically)

        print_score : Bool
        If True, returns the percent of known duplicates that end up placed in the same cluster.

        weight_step : 0
        In short, this indicates at what point the weights from the classifier are added into the embeddings, and does not need to be considered unless you are providing in your own scikit learn pipeline. If you have a pipeline that you are passing in, it is probably the case that you want certain preprocessing steps to take place before skewing the embeddings with the weights. Providing the `weight_step` parameter allows you to index into the scikit pipeline at whichever point you would like and have the embeddings skewed there. See `cluster.fit_predict()` source-code to witness the use of `weight_step` in action
        
        colname: str
        The name of the column added to the `g2v.records` DataFrame, containing the predicted cluster label for each record. Defaults to the value set for `CLUSTER_COL` which can be changed in global_variables.py
        
        Resulting Attibute:

        cluster : Cluster
        Initialized cluster instance with learned clusters stored in `cluster.predicted_labels`.with learned clusters stored in `cluster.predicted_labels`.

        '''

        #No clustering instance or pipeline passed in: Set up default pipeline
        if type(alg) == type(None):
            #If the classifier training examples were standardized, then the embeddings also should also be standardized.
            if self.standardized:
                alg = make_pipeline(StandardScaler(),FeatureAgglomeration(n_clusters=16,affinity='euclidean',linkage='average'),BisectingKMeans(n_clusters=n_clusters,bisecting_strategy='largest_cluster'))
                weight_step = 1
            else:
                alg = make_pipeline(FeatureAgglomeration(n_clusters=16,affinity='euclidean',linkage='average'),BisectingKMeans(n_clusters=n_clusters))
                weight_step = 0
        
        #Need to have `init_embeddings()` called before clustering can take place
        assert hasattr(self,'g2v')
        self.cluster = Clustering(self.g2v,alg,true_labels,self.ratio,weight_step)
        self.cluster.fit_predict(n_clusters=n_clusters,weight_exp=weight_exp,set_attrs=True)
        self.cluster.set_clusters(colname=colname)
        if print_score:
            print(self.cluster.score())


    def init_dedupe(self,n=None,similarity=None,clf=None,training=False,test_size=0.4,ratio=0.1):
        '''init_dedupe(self,similarity=None,clf=None,weight_exp=8,training=False,test_size=0.4,ratio=0.1)
        Following the results of `init_classifer()`, goes through the clusters labeled in `cluster.g2v.records`, computes pairwise distances, and then trains and runs a classifier on said distances in order to predict pairs of duplicates. The resulting predictions are then added to a new column of called `duplicate_col` (which can be set in global_variables.py).

        Inputs:

        n : int
        Number of duplicates to train on. Defaults to training on all possible duplicates.

        similarity : Similarity() instance
        Similarity instance used for creating the distances between pairs

        clf : Scikit learn classifier or pipeline
        Classifier to be trained on for finding inner-cluster duplicates. Needs to support the scikit learn `fit()` and `predict()` interface, a workaround to this for different models is to create a wrapper class with custom designed `fit()` and `predict()` methods.Defaults to Logistic Regression, with StandardScalar (based on whether `standardized` set in `init_classifier()`), see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html and https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html for more information.

        training : Bool
        If true, then the training set will be split (w.r.t `test_size` ratio) and `X_train`,`X_test`,`y_train` and `y_test` attributes will be added

        test_size : float
        If `training` set to True, this will divide the training data into a train set consisting of (1-test_size)% of the data and a test set with test_size % of the data.

        ratio : float
        Ratio of duplicates to non-duplicates to be trained on. If not enough data exists to accomadate the provided ratio, a warning is printed, but the operation still carries on.

        Resulting attributes:

        dedupe : Dedupe() Instance

        df_with_labels : Pandas DataFrame
        DataFrame with clustering and duplicate labels.
        '''

        assert hasattr(self,'cluster')
        cluster_df = self.cluster.g2v.records

        if type(similarity) == type(None):
            similarity = self.sim

        if type(clf) == type(None):
            #Technically, the decision of standardizing here shouldn't be automatically based on the decision for generating the embeddings, but is left anyways for consistency.
            if self.standardized:
                clf = make_pipeline(StandardScaler(),LogisticRegression())
            else:
                clf = make_pipeline(LogisticRegression())

        self.dedupe = Dedupe(cluster_df=cluster_df,clf=clf,similarity=self.sim)
        self.dedupe.count_comparisons(True)
        self.dedupe.get_duplicates(n=n,training=training,test_size=test_size,ratio=ratio)
        self.df_with_labels = self.dedupe.cluster_df


    def save_results(self,encoding='parquet'):
        '''save_results(self,encoding='parquet')
        Writes `df_with_labels` DataFrame (containing learned cluster and duplicate labels) locally to file and folder specified respectively by `output_df_name` and `OUTPUT_PATH`, both found in global_variables.py.

        Input:

        encoding : str
        Specifies the filetype to output. Supports "csv" for csv file or "parquet" for parquet file as options.

        '''

        assert hasattr(self,'dedupe')
        assert hasattr(self,'g2v')
        if not os.path.isdir(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH)

        #Write the DataFrame with cluster and duplicate labels.
        out_path = os.path.join(OUTPUT_PATH,output_df_name)
        if encoding == 'parquet':
            self.df_with_labels.to_parquet(out_path)
        elif encoding == 'csv':
            self.df_with_labels.to_csv(out_path)
        else:
            print('Error: encoding must be one of: "parquet" or "csv". ' + str(encoding) + ' not supported.')
            return

        print(out_path + ' successfully written! ')



                

