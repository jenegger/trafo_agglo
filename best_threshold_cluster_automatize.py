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
##simulated data with peak at 2.1... MeV
my_data = genfromtxt('data_stream_2121.txt', delimiter=',')
#real data from 60Co source
#my_data = genfromtxt('real_data_co60.txt', delimiter=',')
my_data[:,4] = my_data[:,4]+4500  #this step is needed, I only want positive time values, so that I can use the time as a radius

# ### structure of mydata : eventnr, energy, theta, phi, hit-time

my_data = my_data*[1.,1.,3.14159/180,3.14159/180,1.]

num_rows, num_cols = my_data.shape

def sph2cart(az, el, r, energy):
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
        for i in range(0,(len(array_unique_events)-3),3):
                E1 = my_data[my_data[:,0] == array_unique_events[i]]
                E2 = my_data[my_data[:,0] == array_unique_events[i+1]]
                E3 = my_data[my_data[:,0] == array_unique_events[i+2]]
                E_1 = E1[:,[2,3,4,1]]
                E_2 = E2[:,[2,3,4,1]]
                E_3 = E3[:,[2,3,4,1]]
                cart_e1 = sph2cart(E_1[:,1],E_1[:,0],time_weight*E_1[:,2],E_1[:,3])
                np_cart_e1 = np.asarray(cart_e1)
                np_cart_e1 = np.transpose(np_cart_e1)
                
                cart_e2 = sph2cart(E_2[:,1],E_2[:,0],time_weight*E_2[:,2],E_2[:,3])
                np_cart_e2 = np.asarray(cart_e2)
                np_cart_e2 = np.transpose(np_cart_e2)
                
                cart_e3 = sph2cart(E_3[:,1],E_3[:,0],time_weight*E_3[:,2],E_3[:,3])
                np_cart_e3 = np.asarray(cart_e3)
                np_cart_e3 = np.transpose(np_cart_e3)
        
                #create flat arrays for the 3 clusters to compare later with the reconstructed clusters
                arr_energy1 = np_cart_e1[:,3]
                arr_energy1 = arr_energy1.flatten()
                arr_energy2 = np_cart_e2[:,3]
                arr_energy2 = arr_energy2.flatten()
                arr_energy3 = np_cart_e3[:,3]
                arr_energy3 = arr_energy3.flatten()
                list_of_energies = [arr_energy1,arr_energy2,arr_energy3]
                list_before = list_of_energies
                list_of_energies = [l.tolist() for l in list_of_energies]
                list_after = list_of_energies
                #print(list_of_energies)
                all_events += len(list_of_energies)
                anal_list_of_energies = list_of_energies.copy()
                if (len(list_before) != len(list_after)):
                        print("this is list of before:\n")
                        print(list_before)
                        print("this is list of after:\n")
                        print(list_after)
        
                
                X = np.vstack([np_cart_e1,np_cart_e2,np_cart_e3])
                data = pd.DataFrame(X, columns = ['x','y','z','energy'])
                data = data.iloc[:,0:3]
                my_np_data = data.to_numpy()
                
                #clustering_model = AgglomerativeClustering(n_clusters=3, linkage="ward")
                #clustering_model.fit(data)
                output = fclusterdata(data, t=distance_weight, criterion='distance',method="ward")
                #print("this is the output",output)
                #print("this is X:" ,X)
                nr_clusters = len(np.unique(output))    
                array_energy_cluster = np.zeros(nr_clusters)
                arr_reconstructed_clusters = [[] for _ in range(nr_clusters)]
                for i in range(X.shape[0]):
                        array_energy_cluster[output[i]-1] += X[i,3]
                        arr_reconstructed_clusters[output[i]-1].append(X[i,3])
                gamma_energy_distr.append(array_energy_cluster.tolist())
                
                anal_arr_reconstructed_clusters = arr_reconstructed_clusters.copy()
                for j in range(0,len(array_energy_cluster)):
                    if (array_energy_cluster[j] > 1.974 and array_energy_cluster[j] < 2.274):
                        good_counts +=1
                print("thi is the list of energies:")
                for i in range(3): 
                    print(list_of_energies[i])
                print("and this is the list of sorged reco stuff:")
                for i in range(len(arr_reconstructed_clusters)):
                    print(arr_reconstructed_clusters[i])
                for i in range(len(arr_reconstructed_clusters)):
                        for j in range(3):
                                if (sorted(arr_reconstructed_clusters[i]) == sorted(list_of_energies[j])):
                                        well_reconstructed += 1
                                        #remove all well reconstructed energies, by setting value to -1
                                        anal_arr_reconstructed_clusters[i] = [-1]
                                        anal_list_of_energies[j] = [-1]

		####new part creating data with only correctly reco clusters and false negatives
		false_positive = False
                if (len(arr_reconstructed_clusters) >= len(list_of_energies)):
		     for i in range(len(arr_reconstructed_clusters)):
                         for j in range(len(list_of_energies)):
			     if (sorted(arr_reconstructed_clusters[i],reverse=True)[0] ==sorted(list_of_energies[j],reverse=True)[0]:
			         #check for whole lenght of arr_recon.. if same as list_of energies

		if false positive is false - > fill dataset...
		################################################################################
                                        
        
                ##this part is for giving numbers to the wrongly identified clusters
                left_over_list_of_energy = [a  for a in anal_list_of_energies if a[0] != -1]
                left_over_arr_reconstructed_clusters = [a  for a in anal_arr_reconstructed_clusters if a != -1]
                not_reconstructed += len(left_over_list_of_energy)
                #log events which are wrongly reconstructed
                #if (len(left_over_list_of_energy) != 0):
                #    print("-----------------------------------------")
                #    print("this is the event:")
                #    print(X)
                #    print(E1)
                #    print(E2)
                #    print(E3)
                #    print("and this is the reco")
                #    print(arr_reconstructed_clusters)
                #    print("-----------------------------------------")
                print("this is left_over_list_of_energy:")
                print(left_over_list_of_energy)
                print("this is left_over_arr_reconstructed_clusters:")
                print(left_over_arr_reconstructed_clusters)

                for i in range(len(left_over_list_of_energy)):
                        for j in range(len(left_over_arr_reconstructed_clusters)):
                                sorted_loe = sorted(left_over_list_of_energy[i])
                                sorted_lor = sorted(left_over_arr_reconstructed_clusters[j])
                                if (sorted_loe[0] == sorted_lor[0]):
                                        summed_energy = sum(sorted_loe)
                                        for k in range(len(sorted_loe)):
                                                for l in range(len(sorted_lor)):
                                                        if (sorted_loe[k] == sorted_lor[l]):
                                                                sorted_loe[k] = 0
                                                                sorted_lor[l] = 0
                                        summed_false_negative += sum(sorted_loe)/summed_energy
                                        summed_false_positive += sum(sorted_lor)/summed_energy
                ##end of error analysis
        
        ratio_well_reco = well_reconstructed/all_events
        print("ratio of well reconstructed events:\t",ratio_well_reco)
        
        number_for_false_negative = summed_false_negative/not_reconstructed
        number_for_false_positive = summed_false_positive/not_reconstructed
        
        print("key number for false negative:\t", number_for_false_negative)
        print("key number for false positive:\t", number_for_false_positive)
        f = open("output_agglomerative_clustering_threshold_weight_"+str(distance_weight)+"_time_weight_"+str(time_weight)+".txt","a")
        string_to_write = str(distance_weight)+"\t"+str(time_weight)+"\t"+str(ratio_well_reco)+"\n"
        f.write(string_to_write)
        f.close()
        
        my_x = np.hstack(gamma_energy_distr)
        print(my_x)
        plt.xlim([0,7])
        plt.hist(my_x,bins=70)
        plt.yscale('log')
        plt.savefig("agglomerative_clustering_threshold_"+str(distance_weight)+str(time_weight)+"_.png",dpi=300)
        #plt.show()
        print('total number of good counts')
        print(good_counts)
        print('total number of counts')
        print(all_counts)
        print("precentage of correctly identified gammas:", good_counts/all_counts)

if __name__ == "__main__":
        #start = time.time()
        #run_threshold_finding(1520,1)
#       run_threshold_finding(2050)
        #end = time.time()
        #print("TIME OF EXECUTION:\t",end -start)
        run_threshold_finding(3540,5)
        #run_threshold_finding(2800,2)
        #for i in range(1,40):
        #    for j in range(20,4000,20):
        #        process = Process(target=run_threshold_finding,args=(j,i,))
        #        process.start()
#    run_threshold_finding(1520)
#       run_threshold_finding(2050)
#    run_threshold_finding(1750)
#   for i in range(1,100):
#       process = Process(target=run_function,args=(i,))
#       process.start()
