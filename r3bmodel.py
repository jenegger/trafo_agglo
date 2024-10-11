from multiprocessing import Process
from multiprocessing import Pool
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import math
import torch
from numpy import genfromtxt
import sys
np.set_printoptions(threshold=sys.maxsize)

def sph2cart(az, el, r, energy):
    az = (az/180.)*math.pi
    el = (el/180.)*math.pi
    rsin_theta = r*np.sin(el)
    x = rsin_theta*np.cos(az)
    y = rsin_theta*np.sin(az)
    z = r*np.cos(el)
    return x, y, z , energy


def cart2sph(x, y, z, energy):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    th = np.arccos(z/r)
    az = np.arctan2(y,x)
    return th, az, r , energy


###some definitions to calculate the angle
def mag(u, N):
 
    # Stores the final magnitude
    magnitude = 0
 
    # Traverse the array
    for i in range(N):
        magnitude += u[i] * u[i]
 
    # Return the square root of magnitude
    return math.sqrt(magnitude)
 
# Function to find the dot product of the vectors
def dotProd(u, v, N):
 
    # Stores the dot product
    prod = 0
 
    # Traverse the array
    for i in range(N):
        prod = prod + u[i] * v[i]
 
    # Return the product
    return prod
def angleVector(u, v, N):
	#print("this is u:")
	#print(u)
	#print("this is v:")
	#print (v)
	# Stores the dot product of vectors
	dotProductOfVectors = dotProd(u, v, N)
	
	# magnitude of vector u
	magOfu = mag(u, N)
	
	#magnitude of vector v
	magOfv = mag(v, N)
	no_angle_between = 0.01	
	# angle between given vectors
	#print("denominator is:\t", magOfu * magOfv)
	if(magOfu * magOfv == 0):
		return 100000
	elif (dotProductOfVectors/ (magOfu * magOfv) > 0.999):
		return no_angle_between
	else:
		angle = math.acos(dotProductOfVectors
	        / (magOfu * magOfv))
		#return the angle [rad]
		return angle 

def r3bmodel(x,clustersize):
	x = x.numpy()
	list_dist_matrix = []
	for i in range(x.shape[0]):
		data_list = []
		#print("this is the data:")
		#print(x[i,:,:])
		for j in range(x.shape[1]):
			hit = x[i,j,:]
			#print("this is the hit")
			#print(hit)
			hit = hit[[1,2,3,0]]
			#print("this is the hit after moving")
			#print(hit)
			#phi theta time energy
			#print("coordinates r theta phi:\t",hit[2],hit[0],hit[1])
			hit = sph2cart(hit[1],hit[0],hit[2],hit[3])
			#print("this is the hit in xyz:\t",hit)
			#print("x,y,z,t:\t",hit) 
			hit = np.asarray(hit)
			hit = np.transpose(hit)
			data_list.append(hit)
		data = np.asarray(data_list)
		#print("this is the data in xyz coordinates:")
		#print(data)
		#order column
		order_column = np.arange(data.shape[0]).reshape((data.shape[0],1))
		data = np.append(data,order_column,axis=1)
		#cluster number column
		cluster_column = np.full(data.shape[0],-1).reshape((data.shape[0],1))
		data = np.append(data,cluster_column,axis=1)
		orig_data = data.copy()
		#print("orig_data:",orig_data)
		clusternr = 0
		data = data[data[:,3].argsort()[::-1]]
		#print("data",data)
		shape_matrix = data.shape[0]
		foo_list = []
		while (data.shape[0]):
			v_ref = data[0,:]
			v_ref = np.reshape(v_ref,(1,6))
			v_temp = np.empty([0,6])
			#print("this is v_temp:\t",v_temp)
			arr_single_cluster = list()
			for i in range (data.shape[0]):
				angle_ref_hit = angleVector(v_ref[:,0:3].flatten(),(data[i,0:3]).flatten(),3)
				#print("this is reference vector:")
				#print(v_ref[:,0:3].flatten())
				#print("this is data vector:")
				#print((data[i,0:3]).flatten())
				#print("this is calculated angle between vector:\t",angle_ref_hit*180./math.pi)
				mask_orig_data = (orig_data[:,4] == i)
				if ((angle_ref_hit < clustersize)):
					#print("I am inside the clustersize")
					data[i,5] = clusternr
					position = int(data[i,4])
					#orig_data[mask_orig_data,5] = clusternr
					orig_data[position,5] = clusternr
					#print("this is position and clusternr:\t",position,orig_data[position,5])
					foo_list.append([orig_data[mask_orig_data,4],orig_data[mask_orig_data,5]])
				elif (np.all(v_ref[:,:].flatten() == data[i,:].flatten())):
					#print("I am simply comparing myself....")
					data[i,5] = clusternr
					position = int(data[i,4])
					#orig_data[mask_orig_data,5] = clusternr
					orig_data[position,5] = clusternr
					foo_list.append([orig_data[mask_orig_data,4],orig_data[mask_orig_data,5]])
				else :
					#print("I am outside the cluster")
					v_temp = np.append(v_temp,np.array(data[i,:]))
			clusternr += 1
			v_temp = np.reshape(v_temp,(-1,6))
			data = v_temp
		#print("this is final orig_data",orig_data)
		matrix = np.zeros([shape_matrix,shape_matrix],dtype=float)
		for i in range(shape_matrix):
			for j in range(shape_matrix):
				if (orig_data[i,5] == orig_data[j,5]):
					matrix[int(orig_data[i,4]),int(orig_data[j,4])] = 1
				else :
					matrix[int(orig_data[i,4]),int(orig_data[j,4])] = 0
		matrix = np.triu(matrix,1)
		#print("THIS IS THE MATRIX")
		#print(matrix)
		list_dist_matrix.append(matrix)
	np_array_dist_matrix = np.asarray(list_dist_matrix)
	torch_matrix = torch.from_numpy(np_array_dist_matrix)
	upper_tri_mask = torch.triu(torch.ones((torch_matrix.shape[1],torch_matrix.shape[1])),diagonal=1).bool()
	#print("this is the datatype of r3bmodel return value:\t",type(matrix[0,0]))
	return torch_matrix[:,upper_tri_mask]
