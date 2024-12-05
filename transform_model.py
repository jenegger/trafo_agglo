#!/usr/bin/env python3
import math
import torch
from torch import nn
import torch.nn.functional as F
from itertools import combinations

class transformer_model(nn.Module):
	def __init__(self, k, heads=8, mask=False):
		super().__init__()
		assert k % heads == 0
		self.k, self.heads = k, heads
		self.tokeys    = nn.Linear(k, k, bias=False)
		self.toqueries = nn.Linear(k, k, bias=False)
		self.tovalues  = nn.Linear(k, k, bias=False)
		self.unifyheads = nn.Linear(k, k)
		self.linear_embedding = torch.nn.Linear(4,k)

	def forward(self, x,in_hitnr):
		x = self.linear_embedding(x)	
		b,t, k = x.size()
		h = self.heads
		queries = self.toqueries(x)
		keys    = self.tokeys(x)
		values  = self.tovalues(x)
		s = k // h
		keys    = keys.view(b,t, h, s)
		queries = queries.view(b,t, h, s)
		values  = values.view(b,t, h, s)
		# - fold heads into the batch dimension
		keys = keys.transpose(1, 2).contiguous().view(b*h, t, s)
		queries = queries.transpose(1, 2).contiguous().view(b*h, t, s)
		values = values.transpose(1, 2).contiguous().view(b*h, t, s)
		# Get dot product of queries and keys, and scale
		dot = torch.bmm(queries, keys.transpose(1, 2))	
		dot = dot / (s ** (1/2))
		#dot = F.softmax(dot,dim=2)
		out = torch.bmm(dot, values).view(b,h, t, s) 
		out = out.transpose(1,2).contiguous().view(b,t, s * h)
		out = self.unifyheads(out)
		#use cosine similarity
		L2_dist = torch.cosine_similarity(out[:,None] , out[:,:,None],dim=-1)		
		L2_dist = 0.5*(L2_dist+1)
		upper_tri_mask = torch.triu(torch.ones((out.shape[1],out.shape[1])),diagonal=1).bool() #out[1] is max hit number in batch 

		return L2_dist[:,upper_tri_mask]




class transformer_model_extended(nn.Module):
	def __init__(self, features, heads=8, mask=False):
		super().__init__()
		self.features, self.heads = features, heads
		self.linear_embedding = torch.nn.Linear(4,self.features)
		self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.features, nhead=self.heads)
		self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
		#self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=8)
		self.activation = torch.nn.ReLU()
		self.additional_linear_layer = torch.nn.Linear(self.features,self.features)
		

	def forward(self, x,in_hitnr):
		x = self.linear_embedding(x)	
		#x = self.activation(x)  #put in some relu function to test....	
		#x = self.additional_linear_layer(x) #additional linear layer
		out = self.transformer_encoder(x)	
		#out = self.additional_linear_layer(out)
		#out = self.activation(out)
		#out = self.additional_linear_layer(out)
		#use cosine similarity
		#print("this is the shape of out:\t", out.shape)
		L2_dist = torch.cosine_similarity(out[:,None] , out[:,:,None],dim=-1)		
		L2_dist = 0.5*(L2_dist+1)
		#print("this is the shape of L2_dist:\t", L2_dist.shape)
		upper_tri_mask = torch.triu(torch.ones((out.shape[1],out.shape[1])),diagonal=1).bool() #out[1] is max hit number in batch 
		ret_val = L2_dist[:,upper_tri_mask]
		#out_ret_val = torch.where(ret_val > 0.7, torch.tensor(1), torch.tensor(0)).float()
		out_ret_val = torch.where(ret_val > 0.7, torch.tensor(1,requires_grad=False), torch.tensor(0,requires_grad=False)).float()
		#out_ret_val = (ret_val > 0.8).float()
		#out_ret_val = torch.tensor(out_ret_val, requires_grad=True)
		#out_ret_val = torch.where(ret_val > 0.7, torch.FloatTensor(1,requires_grad=True), torch.FloatTensor(0,requires_grad=True))


		###tensor_i = out.unsqueeze(2)  # shape: (64, 10, 1, 32)
		###tensor_j = out.unsqueeze(1)  # shape: (64, 1, 10, 32)
		###expansion_factor = out.shape[1]
		###output_tensor = torch.cat([tensor_i.expand(-1, -1, expansion_factor, -1), tensor_j.expand(-1, expansion_factor, -1, -1)], dim=-1)
		####small net:
		###net = nn.Sequential(
      		###	nn.Linear(64,8),
		###	nn.ReLU(),
		###	nn.Linear(8,1),
		###	nn.Sigmoid()
		###	)
		###res = net(output_tensor)
		###temp_res = torch.squeeze(res)
		###upper_tri_mask = torch.triu(torch.ones((temp_res.shape[1],temp_res.shape[1])),diagonal=1).bool() #out[1] is max hit number in batch 
		###result = temp_res[:,upper_tri_mask]

		###


		###return result
		#return ret_val #--> as I have done before....
		return ret_val



class feed_forward_model(nn.Module):
	def __init__(self, features, heads=8, mask=False):
		super().__init__()
		self.linear = torch.nn.Linear(8,100)
		self.another_linear = torch.nn.Linear(100,100)
		self.activation = torch.nn.ReLU()
		self.linear_back = torch.nn.Linear(100,1)

	def forward(self, x,in_hitnr):
		num_vectors = x.shape[1]
		batch_size = x.shape[0]
		pair_indices = list(combinations(range(num_vectors), 2))
		num_pairs = len(pair_indices)
		output_tensor = torch.zeros(batch_size, num_pairs, 8)
		for i, (idx1, idx2) in enumerate(pair_indices):
		# Add the vectors from the input tensor based on pair indices
			combined_features = torch.cat((x[:, idx1], x[:, idx2]), dim=-1)
			output_tensor[:, i] = combined_features
		print("the shape of output tensor:")
		print(output_tensor.shape)
		output_tensor = self.linear(output_tensor)
		output_tensor = self.another_linear(output_tensor) 
		output_tensor = self.activation(output_tensor)
		output_tensor = self.linear_back(output_tensor)
		output_tensor = torch.sigmoid(output_tensor)
		output_tensor = torch.squeeze(output_tensor)
		return output_tensor

		













