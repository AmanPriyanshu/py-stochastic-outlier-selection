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

def get_perplexity(D_row, variance):
    	A_row = np.exp(-D_row * variance)
    	sumA = sum(A_row)
    	perplexity = np.log(sumA) + variance * np.sum(D_row * A_row) / sumA
    	return perplexity, A_row

def affinity_matrix(dMatrix, perplexity):
	(n, _) = dMatrix.shape
	variance_matrix = np.ones(dMatrix.shape[0])
	affinity_matrix = np.zeros(dMatrix.shape)
  	logU = np.log(perplexity)
	for i in range(dMatrix.shape[0]):
		variance_min = -np.inf
	    	variance_max =  np.inf
	    	d_i = dMatrix[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
	    	(c_perplexity, thisA) = get_perplexity(d_i, variance_matrix[i])
	    	perplexity_diff = c_perplexity - logU
	    	tries = 0
	    	while (np.isnan(perplexity_diff) or np.abs(perplexity_diff) > eps) and tries < 5000:
	      		if np.isnan(perplexity_diff):
				variance_matrix[i] = variance_matrix[i] / 10.0
	      		elif perplexity_diff > 0:
				variance_min = variance_matrix[i].copy()
				if variance_max == np.inf or variance_max == -np.inf:
		  			variance_matrix[i] = variance_matrix[i] * 2.0
				else:
		  			variance_matrix[i] = (variance_matrix[i] + variance_max) / 2.0
	      		else:
				variance_max = variance_matrix[i].copy()
				if variance_min == np.inf or variance_min == -np.inf:
		  			variance_matrix[i] = variance_matrix[i] / 2.0
				else:
		  		variance_matrix[i] = (variance_matrix[i] + variance_min) / 2.0
	      		(c_perplexity, thisA) = get_perplexity(d_i, variance_matrix[i])
	      		perplexity_diff = c_perplexity - logU
	      		tries += 1
	    	affinity_matrix[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisA
	return variance_matrix, affinity_matrix
	

def binding_matrix(aMatrix):
	binding_matrix = aMatrix / aMatrix.sum(axis=1)[:,np.newaxis]
	return binding_matrix


def outlier_probability(bMatrix):
	outlier_matrix = np.prod(1-bMatrix, 0)
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






print(outlier_matrix)                     #This matrix is the probability whether a point is an outlier or not as we see I have manipulated the 
#first data to all values being 0 and hence it has a probability of 0.95265683 and hence is definitely an outlier


