import os
import pandas as pd
import numpy as np
from math import exp

def data_reader(path, axis):
	dataset = pd.read_csv(path)
	initial_data_shape = dataset.shape
	dataset.dropna(axis=axis, how='any', thresh=None, subset=None, inplace=False) #https://hackersandslackers.com/pandas-dataframe-drop/
	final_data_shape = dataset.shape
	return dataset.to_dict(), (initial_data_shape, final_data_shape)

def feature_extractor(dataset, metric_columns):
	final_data = {}
	for i in metric_columns:
		final_data.update({i:dataset[i]})
	return final_data


def data_cleaner_numerically(data, remove_nan=False, features_to_remove=[], shape=[]):
	rows_to_remove = []
	for i in features_to_remove:
		feature = list(data[i].values())
		for j in range(len(feature)):
			if np.isnan(feature[j]) and j not in rows_to_remove:
				rows_to_remove.append(j)
	cleaned_data = np.empty([shape[0]-len(rows_to_remove), len(data.keys())])
	for i,j in zip(data,range(len(data.keys()))):
		values = list(data[i].values())
		c = 0
		for k in range(len(values)):
			if k not in rows_to_remove:
				cleaned_data[c][j] = values[k]/max(values)
				c += 1

	return cleaned_data
			

def dissimilarity_matrix(data):
	dissimilarity_matrix = np.empty([data.shape[0], data.shape[0]])
	for i in range(len(data)):
		for j in range(len(data)):
			d_ij = np.sum(np.square(data[i]-data[j]))
			dissimilarity_matrix[i][j] = d_ij

	return dissimilarity_matrix

def affinity_matrix(dMatrix, perplexity):
	variance_matrix = np.empty(dMatrix.shape[0])
	affinity_matrix = np.empty(dMatrix.shape)
	for i in range(dMatrix.shape[0]):
		variance = 0.0001
		tempPerplexity = 0
		while tempPerplexity <= perplexity:
			a_i = [0 if i==j else exp(-(dMatrix[i][j]/variance)**2) for j in range(dMatrix.shape[1])]
			tempPerplexity = sum(a_i)
			variance += 0.001
		variance_matrix[i] = variance

	for i,variance in zip(range(dMatrix.shape[0]), variance_matrix):
		for j in range(dMatrix.shape[1]):
			affinity_matrix[i][j] = exp(-(dMatrix[i][j]/variance)**2)
	return variance_matrix, affinity_matrix

	

def binding_matrix(aMatrix):
	binding_matrix = np.empty(aMatrix.shape)
	for i in range(aMatrix.shape[0]):
		for j in range(aMatrix.shape[1]):
			binding_matrix[i][j] = aMatrix[i][j]/np.sum(aMatrix[i])

	return binding_matrix


def outlier_probability(bMatrix):
	outlier_matrix = np.empty(bMatrix.shape[0])
	for i in range(bMatrix.shape[0]):
		condition = [(1-bMatrix[j][i]) for j in range(bMatrix.shape[0])]
		outlier_matrix[i] = np.prod(condition)
	return outlier_matrix



dataset, shape_change_after_cleaning = data_reader('./cardataset.csv', 0)
data = feature_extractor(dataset, ["Engine HP", "Engine Cylinders", "highway MPG", "city mpg", "Popularity", "MSRP"])
cleaned_data = data_cleaner_numerically(data, remove_nan=True, features_to_remove=["Engine HP", "Engine Cylinders", "highway MPG", "city mpg", "Popularity", "MSRP"], shape=shape_change_after_cleaning[1])
dMatrix = dissimilarity_matrix(cleaned_data[:500])
variance_matrix, affinity_matrix = affinity_matrix(dMatrix, 50)
binding_matrix = binding_matrix(affinity_matrix)
outlier_matrix = outlier_probability(binding_matrix)



# Now here the binding matrix basically gives the probability of a particular
#print(binding_matrix[41][2])    #7.009510987249058e-13
#print(binding_matrix[3][2])     #0.015065941745409707






print(outlier_matrix)                     #This matrix is the probability whether a point is an outlier or not
#Now here we can clearly see that the 41st index is an outlier compared to the 2 and 3. This dataset was not meant for outlier detection however, this is how it would work. All references are taken from the paper: