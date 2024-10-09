from multiprocessing import Process
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import time
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from numpy import genfromtxt
from scipy.cluster.hierarchy import fclusterdata
my_data = genfromtxt('data_stream_2121.txt', delimiter=',')
#just make positive time in data
my_data[:,4] = my_data[:,4]+4500
### structure of mydata : eventnr, energy, theta, phi, hit-time
my_data = my_data*[1.,1.,3.14159/180,3.14159/180,1.]

num_rows, num_cols = my_data.shape


def uni_sph2cart(eventnr,energy,az, el,time):
	r = 1
	rsin_theta = r*np.sin(el)
	x = rsin_theta*np.cos(az)
	y = rsin_theta*np.sin(az)
	z = r*np.cos(el)
	return np.array([eventnr,x,y,z,energy,time])

def uni_cart2sph(eventnr,x, y, z, energy,time):
	hxy = np.hypot(x, y)
	r = np.hypot(hxy, z)
	th = np.arccos(z/r)
	az = np.arctan2(y,x)
	return np.array([eventnr,energy,th,az,time])

##this definitions are used when using time as radius
def sph2cart(energy,az, el,time):
	r = time
	rsin_theta = r*np.sin(el)
	x = rsin_theta*np.cos(az)
	y = rsin_theta*np.sin(az)
	z = r*np.cos(el)
	return np.array([x,y,z,energy])

def cart2sph(x, y, z, energy):
	hxy = np.hypot(x, y)
	r = np.hypot(hxy, z)
	th = np.arccos(z/r)
	az = np.arctan2(y,x)
	return np.array([th,az,r,energy])

def func_cm(data):
	print(data[:,0])
	assert np.all(data[:, 0] == data[0, 0]) == True
	positions = data[:,1:4]
	masses = data[:,4]
	cm = np.sum(positions.T * masses, axis=1) / np.sum(masses)
	time = data[np.argmax(data[:,5],axis=0),5]
	return np.array([data[0,0],cm[0],cm[1],cm[2],np.sum(masses),time])

array_unique_events = np.unique(my_data[:,0])
arr_cluster_nr = []
gamma_energy_distr = []
def run_threshold_finding(distance_weight,time_weight):
	time_weight = time_weight/2.
	all_events = 0.
	well_reconstructed = 0.
	good_counts = 0
	not_reconstructed = 0.
	summed_false_negative = 0.
	summed_false_positive = 0.      
	all_counts = len(array_unique_events)
	j = 0
	#for i in range(0,(len(array_unique_events)-3),3):
	for i in range(0,3,3):
		E1 = my_data[my_data[:,0] == array_unique_events[i]]
		E2 = my_data[my_data[:,0] == array_unique_events[i+1]]
		E3 = my_data[my_data[:,0] == array_unique_events[i+2]]
		cart_e1 = np.transpose(sph2cart(E1[:,1],E1[:,2],E1[:,3],E1[:,4]))
		cart_e2 = np.transpose(sph2cart(E2[:,1],E2[:,2],E2[:,3],E2[:,4]))
		cart_e3 = np.transpose(sph2cart(E3[:,1],E3[:,2],E3[:,3],E3[:,4]))
		#print(cart_e1)
		data = pd.DataFrame(np.vstack([cart_e1,cart_e2,cart_e3]), columns = ['x','y','z','energy'])
		output = fclusterdata(data, t=distance_weight, criterion='distance',method="ward")
		##selection criteria- TODO: add
		E1[:,0] = j
		E2[:,0] = j+1
		E3[:,0] = j+2
		j += 3
		full_cart1 = np.transpose(uni_sph2cart(E1[:,0],E1[:,1],E1[:,2],E1[:,3],E1[:,4])) 
		full_cart2 = np.transpose(uni_sph2cart(E2[:,0],E2[:,1],E2[:,2],E2[:,3],E2[:,4])) 
		full_cart3 = np.transpose(uni_sph2cart(E3[:,0],E3[:,1],E3[:,2],E3[:,3],E3[:,4])) 
		full_events = np.vstack([full_cart1,full_cart2,full_cart3])
		output = np.reshape(output,(-1,1))	
		full_events_cluster = np.append(full_events,output,axis=1)
		summarized_clusters_list = []
		print(full_events_cluster)
		while full_events_cluster.shape[0]:
			clusternr = full_events_cluster[0][-1]
			temp_cluster_hits = full_events_cluster[full_events_cluster[:,-1] == clusternr]
			cm_cluster = func_cm(temp_cluster_hits)
			print(cm_cluster.shape)
			cm_cluster_sph = uni_cart2sph(cm_cluster[0],cm_cluster[1],cm_cluster[2],cm_cluster[3],cm_cluster[4],cm_cluster[5]) #todo check that indexing goes right
			summarized_clusters_list.append(cm_cluster_sph)	
			full_events_cluster = full_events_cluster[full_events_cluster[:,-1] != clusternr]			
		print(summarized_clusters_list)
			
				




if __name__ == "__main__":
	run_threshold_finding(3540,5)






