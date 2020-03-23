#!/usr/bin/python

# Try to open imports
try:
    import sys
    import random
    import math
    import os
    import time
    import numpy as np
	import pandas as pd
    from matplotlib import pyplot as plt
	from sklearn import datasets
	from sklearn.model_selection import train_test_split
	from sklearn.neighbors import NearestNeighbors
	from sklearn.neighbors import KDTree
	from sklearn.datasets import load_iris
	from sklearn.model_selection import train_test_split
	from sklearn.naive_bayes import GaussianNB
	from sklearn.metrics import confusion_matrix
	from sklearn.decompostion import PCA as sklearnPCA
	from sklearn.model_selection import cross_val_score
	# Heat Map
	import seaborn as sns

# Error when importing
except ImportError:
    print('### ', ImportError, ' ###')
    # Exit program
    exit()


# Read input
def read():
	# Week 3
	# Classification
	#digits
    digits = datasets.load_digits()	
	X = digits.datasets
	y = digits.target
	# Train
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
	train_test_split(y, shuffle = False)
	# N Nearest Neighbors
	nbrs = NearestNeighbors(n_neighbors = 2, algorithm = 'ball_tree').fit(X)
	distances = nbrs.kneighbors(X)
	indices = nbrs.kneighbors(X)
    kdt = KDTree(X, leaf_size = 30, metric = 'euclidean')
	kdt.query(X, k  2, return_distance = False)
	# Naive Bayes
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)
	gnb = GaussianNB()
	y_pred = gnb.fit(X_train, y_train).predict(X_test)
	print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))
    # Regresion
	np.random.seed(0)
	sns.set()
	pd.read_csv('creditcard.csv')  
    data = pd.read_csv('gapminder.csv')
    y = data[4]
    X = data.iloc[:, 0:4]
	# Heat Map
	uniform_data = np.random.rand(10, 12)
	ax = sns.heatmap(uniform_data)
	credit = sns.load_dataset("creditcard.csv")
	# linear Regresion
	sns.regplot(x  "Fertility", y = "Life Expectancy", data = credits);
    fruads = df.loc[df['Class'] == 1]
	non_fruads = df.loc[df['Class'] == 0]
	print(len(frauds), 'frauds, ', len(non_fruads), 'nonfrauds')
# Main
def main():
    # Read Input
    read()
    # Close Program
    exit()


# init
if __name__ == '__main__':
    # Begin
    main()


