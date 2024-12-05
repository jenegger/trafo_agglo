#!/usr/bin/env python3
import itertools
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import math
from numpy import genfromtxt

def create_data():
	#read from file
	my_data = genfromtxt('data_stream_2121.txt', delimiter=',')
	#my_data = genfromtxt('file.txt', delimiter=',')
	#my_data = genfromtxt('small_file.txt', delimiter=',')
	#my_data = genfromtxt('test_output_false_negative.txt', delimiter=',')
	#normalize data
	my_data[:,4] =(my_data[:,4]-np.min(my_data[:,4]))/(np.max(my_data[:,4])-np.min(my_data[:,4]))
	#my_data[:,1] =(my_data[:,1]-np.min(my_data[:,1]))/(np.max(my_data[:,1])-np.min(my_data[:,1]))
	array_unique_events = np.unique(my_data[:,0])
	size_of_unique_events = array_unique_events.shape[0];
	modulo_val = size_of_unique_events % 3
	if (modulo_val == 0):
		print("modulo is 0")
	if (modulo_val == 1):
		print("modulo is 1")
		array_unique_events = array_unique_events[:-1]
	if (modulo_val == 2):
		print("modulo is 2")
		array_unique_events = array_unique_events[:-2]
	selected_hits = np.empty(0)
	eventnumber = array_unique_events.shape[0]/3
	#selected_hits = np.random.choice(array_unique_events,3*int(eventnumber),replace=False)
	#IMPORTANT: do not randomize events!!!
	#print(array_unique_events)
	selected_hits = array_unique_events.reshape((-1,3))
	#print(selected_hits)
	#selected_hits = np.resize(selected_hits,(int(eventnumber),3))
	data_list =  []
	mask_list = []
	hitnr_list = []
	#for hits in selected_hits[:256]:
	for hits in selected_hits:
		nr_subevent1 = int(hits[0])
		nr_subevent2 = int(hits[1])
		nr_subevent3 = int(hits[2])
		subevent1 = my_data[my_data[:,0] == nr_subevent1]
		subevent2 = my_data[my_data[:,0] == nr_subevent2]
		subevent3 = my_data[my_data[:,0] == nr_subevent3]
		full_event = np.concatenate((subevent1,subevent2,subevent3),axis=0)
		np.random.shuffle(full_event)
		size_corr_matrix = full_event.shape[0]
		corr_matrix = np.zeros((size_corr_matrix,size_corr_matrix))
		for i in range(size_corr_matrix):
			for j in range(size_corr_matrix):
				if (full_event[i,0] == full_event[j,0]):
					corr_matrix[i,j] = 1.
		corr_matrix = np.triu(corr_matrix,+1)	
		full_event = np.delete(full_event,0,1)
		data_list.append(full_event)
		mask_list.append(corr_matrix)
		hitnr_list.append(np.int64(corr_matrix.shape[0]))
	
	return data_list, mask_list,hitnr_list
