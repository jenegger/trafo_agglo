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


def uni_sph2cart(energy,az, el,time):
	r = 1
	rsin_theta = r*np.sin(el)
	x = rsin_theta*np.cos(az)
	y = rsin_theta*np.sin(az)
	z = r*np.cos(el)
	return np.array([x,y,z,energy,time])

def uni_cart2sph(x, y, z, energy,time):
	hxy = np.hypot(x, y)
	r = np.hypot(hxy, z)
	th = np.arccos(z/r)
	az = np.arctan2(y,x)
	return np.array([energy,th,az,time])

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
	#for i in range(0,(len(array_unique_events)-3),3):
	for i in range(5):
		E1 = my_data[my_data[:,0] == array_unique_events[i]]
		E2 = my_data[my_data[:,0] == array_unique_events[i+1]]
		E3 = my_data[my_data[:,0] == array_unique_events[i+2]]
		print("energy,az, el,time")
		print("this is in  spherical coordinates:\t",E1)		
		cart_e1 = np.transpose(sph2cart(E1[:,1],E1[:,2],E1[:,3],E1[:,4]))
		#print("[x,y,z,energy]")
		#print ("this is in cart:\t", cart_e1)
		cart_e2 = np.transpose(sph2cart(E2[:,1],E2[:,2],E2[:,3],E2[:,4]))
		cart_e3 = np.transpose(sph2cart(E3[:,1],E3[:,2],E3[:,3],E3[:,4]))
		#print(cart_e1)
		data = pd.DataFrame(np.vstack([cart_e1,cart_e2,cart_e3]), columns = ['x','y','z','energy'])
		output = fclusterdata(data, t=distance_weight, criterion='distance',method="ward")
		print (output)



if __name__ == "__main__":
	run_threshold_finding(3540,5)






