import networkx as nx
def energy_clusters(comb_tensor,data_tensor,cut_val):
	#print ("comb_tensor:\t",comb_tensor)
	#print ("data_tensor:\t",data_tensor)
	clusters_list  = []		
	index = 0
	n = data_tensor.shape[0]
	#print("this is n:\t",n)
	#print("this is shape of data:\t" , data_tensor.shape)
	for i in range(0,n):
		cluster = []
		for j in range(i,n):
			#print("this is i  and j :\t", i,j)
			if (i == j):
				#print("i is equal j")
				clusters_list.append((i,j))
			elif(comb_tensor[index] > cut_val and i != j):
				#print(" i is not equal j and comb tensor is 1")
				clusters_list.append((i,j))
				index += 1
			else:
				#print(" i is not equal j and comb tensor is 0")
				index += 1
		#clusters_list.append(cluster)
	#print("this is clusters_list before using networkx")
	#print(clusters_list)
	#print("this is the type of the clusters_list\t", type(clusters_list))
	G=nx.Graph()
	G.add_edges_from(clusters_list)
	final_cluster_list = list(nx.connected_components(G))
	final_energy_clusters = []
	for i in final_cluster_list:
		subev_cluster = i
		subev_energy = 0
		for j in subev_cluster:
			if (data_tensor[j,0] > 0):
				subev_energy += data_tensor[j,0].item()
		if (subev_energy > 0):
			#print("this is cluster energy:\t", subev_energy)
			final_energy_clusters.append(subev_energy)
	return final_energy_clusters
			
