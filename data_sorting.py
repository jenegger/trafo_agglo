import torch
import networkx as nx
import sys
import math
import statistics
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from data_shaping import create_data
from transform_model import transformer_model
from transform_model import transformer_model_extended
from transform_model import feed_forward_model
from matplotlib import pyplot as plt
from r3bmodel import r3bmodel
from idea import energy_clusters


class CustomDataset(Dataset):
    def __init__(self):
        self.d_set, self.target,self.hit_nr = create_data()
        assert  len(self.d_set) == len(self.target)

    def __len__(self):
        return len(self.d_set)

    def __getitem__(self, idx):
        event = self.d_set[idx]
        mask = self.target[idx]
        hit_nr = self.hit_nr[idx]
        return event,mask,hit_nr

def dynamic_length_collate(batch):
	list_of_lists = list(map(list, batch))
	in_data = [sublist[0] for sublist in list_of_lists]
	in_target = [sublist[1] for sublist in list_of_lists]
	in_hitnr = [sublist[2] for sublist in list_of_lists]
	nr_max_hits = max(in_hitnr)
	out_data = []
	out_target = []
	out_hitnr = []
	for in_data,in_target,in_hitnr in batch:
		pad_nr = nr_max_hits - in_hitnr
		zero_array_data = np.zeros((pad_nr,in_data.shape[1]))
		result_data = np.concatenate((in_data, zero_array_data), axis=0)
		zero_array_target = np.zeros((nr_max_hits,nr_max_hits))
		result_target = np.pad(in_target, ((0, nr_max_hits - in_target.shape[0]), (0, nr_max_hits - in_target.shape[1])), mode='constant', constant_values=0)		
		out_data.append(result_data)
		out_target.append(result_target)
		out_hitnr.append(in_hitnr)
	np_out_data = np.array(out_data)
	np_out_target = np.array(out_target)
	np_out_hitnr = np.array(out_hitnr)
	t_out_data = torch.from_numpy(np_out_data).float()
	t_out_target = torch.from_numpy(np_out_target).float()
	t_out_hitnr = torch.from_numpy(np_out_hitnr).float()
	return t_out_data,t_out_target,t_out_hitnr
		
		


bs = 64
dataset = CustomDataset()
dloader = DataLoader(dataset,batch_size=bs,shuffle=False,collate_fn=dynamic_length_collate)
#epoch-iteration infos ----
n_epochs = 10
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/bs)
print (total_samples,n_iterations)
#--------------------------

# Train the model
model = "feed_forward"  #there are the "homemade" model and the "pytorch_model" and "feed_forward"
lf = "bce_only" #there are the options "logit" and "bce_only"

dtype = torch.float32
feature_nr = 32
#loss_rate = 2e-4
loss_rate = 2e-4

if (model == "homemade"):
	transformer_model = transformer_model(feature_nr)
if (model == "pytorch_model"):
	transformer_model = transformer_model_extended(feature_nr)
if (model == "feed_forward"):
	transformer_model = feed_forward_model(feature_nr)
if (lf == "logit"):
	loss_fn = nn.BCEWithLogitsLoss()
if (lf == "bce_only"):
	loss_fn = nn.BCELoss()
	
if (model == "homemade" or model == "pytorch_model" or model == "feed_forward"):
	optimizer = optim.SGD(transformer_model.parameters(),lr=loss_rate)
	transformer_model.train()
l_loss = []
for epoch in range(n_epochs):
	#for X_batch,target,in_hitnr in dloader:
	for i,(X_batch,target,in_hitnr) in enumerate(dloader):
		if (i+1) % 50 == 0:
			print(f"epoch {epoch+1}/{n_epochs}, step {i+1}/{n_iterations}")
		if (model == "homemade"):
			y_pred = transformer_model(X_batch,in_hitnr)
			y_true = target[:in_hitnr,:in_hitnr]
		if (model == "pytorch_model" or model == "feed_forward"):
			print("this is the shape of the input:")
			print(X_batch.shape)
			y_pred = transformer_model(X_batch,in_hitnr)
			torch.set_printoptions(threshold=10000)
			y_true = target
			print("shape of y predicted:",y_pred.shape)
			print("shape of y true:",y_true.shape)
		if (model == "r3bmodel"):
			y_pred = r3bmodel(X_batch,0.25).float()
			y_true = target.float()
		upper_tri_mask = torch.triu(torch.ones(((torch.max(in_hitnr)).type(torch.int64),(torch.max(in_hitnr).type(torch.int64)))),diagonal=1).bool()
		y_true = y_true[:,upper_tri_mask]
		#if (epoch == 9):
			#print("true values:")
			#print(y_true[1,:])
			#print("pred_values:")
			#print(y_pred[1,:])
		##insert checks for energy spectum
		#print ("this is y predicted:")
		#print(y_pred.shape)
		#print("thsi is X_batch shape:\t",X_batch.shape)
		#data_test = X_batch[0,:,:]
		#comb_test = y_pred[0,:]
		#idea_test = energy_clusters(comb_test,data_test)
		#print(idea_test)
		#reco_list.append(idea_test)

		##end of checks
		loss  = loss_fn(y_pred,y_true)
		if (model == "homemade" or model == "pytorch_model" or model == "feed_forward"):
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		l_loss.append(loss.item())
		if (model == "homemade" or model == "pytorch_model" or model == "feed_forward"):
			torch.save(transformer_model,'model_scripted.pt') # Save

#mean_loss = statistics.mean(l_loss[1500:])
#print("this is mean losss over all epochs and steps:\t",mean_loss)

plt.title("Loss functions")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.plot(l_loss)
plt.show()
#plt.savefig("loss_func_r3bmodel_bs16_bce.png",dpi=300)


###new part
import itertools
#merged = list(itertools.chain.from_iterable(reco_list))
#plt.hist(merged,bins=100)
#plt.show()
#here I evaluate the model
if (model == "homemade" or model == "pytorch_model" or model == "feed_forward"):
	with torch.no_grad():
		transformer_model = torch.load('model_scripted.pt')
		transformer_model.eval()
#for cut in range(500,1000,25):
for cut in range(200,700,25): #this is for the feed forward method, since no values above 0.5
#for cut in range(500,501):
	cut = cut/1000.
	reco_list = []
	true_list = []
	single_hit_list = []
	class1 = []
	class0 = []
	for i,(X_batch,target,in_hitnr) in enumerate(dloader):

		if (model == "homemade"):
			y_pred = transformer_model(X_batch,in_hitnr)
			y_true = target[:in_hitnr,:in_hitnr]
		if (model == "pytorch_model" or model == "feed_forward"):
			y_pred = transformer_model(X_batch,in_hitnr)
			torch.set_printoptions(threshold=10000)
			y_true = target
			print("y_pred")
			print(y_pred)
		if (model == "r3bmodel"):
			y_pred = r3bmodel(X_batch,0.25).float()
			y_true = target.float()
		upper_tri_mask = torch.triu(torch.ones(((torch.max(in_hitnr)).type(torch.int64),(torch.max(in_hitnr).type(torch.int64)))),diagonal=1).bool()
		y_true = y_true[:,upper_tri_mask]
		y_single_hit = target
		zero_mask = torch.zeros((torch.max(in_hitnr)).type(torch.int64),(torch.max(in_hitnr)).type(torch.int64)).bool()
		y_single_hit = y_single_hit[:,zero_mask]
		y_single_hit = []
		for l in range(X_batch.shape[0]):	
			data_test = X_batch[l,:,:]
			comb_test = y_pred[l,:]
			idea_test = energy_clusters(comb_test,data_test,cut)
			true_test = energy_clusters(y_true[l,:],data_test,cut)
			y_single_hit = torch.zeros(y_true[l,:].shape)
			single_hit_test = energy_clusters(y_single_hit,data_test,cut)
			reco_list.append(idea_test)
			true_list.append(true_test)
			single_hit_list.append(single_hit_test)
			for m in range(y_true[l,:].shape[0]):
				classes = y_true[l,:]
				if classes[m] == 1. :
					class1.append(comb_test[m].detach().item())
				else:
					class0.append(comb_test[m].detach().item())	
	merged = list(itertools.chain.from_iterable(reco_list))
	merged_true = list(itertools.chain.from_iterable(true_list))
	merged_single_hit = list(itertools.chain.from_iterable(single_hit_list))
	plt.hist(merged,bins=100,range=(0,8),label=model,color="green",alpha=0.5)
	plt.hist(merged_true,bins=100,range=(0,8),label="true",color="black",alpha=0.5)
	plt.hist(merged_single_hit,bins=100,range=(0,8),label="single clusters",facecolor='none',edgecolor = "red",linewidth=0.5)
	plt.legend()
	titlename = "cutting edge at" + str(cut)
	plt.title(titlename)
	plt.yscale('log')
	plot_name = str(model)+str("_")+ str(n_epochs) + str("_") +str(cut)+str(".png")
	plt.savefig(plot_name,dpi=300)
	plt.clf()

	list_class1 = list(class1)
	list_class0 = list(class0)
	plt.hist(list_class1,bins=100,range=(0,1),label="belonging together",color="red",alpha=0.5)
	plt.hist(list_class0,bins=100,range=(0,1),label="independent",color="blue",alpha=0.5)
	plt.legend()
	plt.savefig("class_distributions.png")
	plt.clf()
	#plt.show()
