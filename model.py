import torch
import torch.nn as nn
import numpy as np
import tqdm

class MLP(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
            super(MLP, self).__init__()
            self.input_size = input_size
            self.hidden_size1  = hidden_size1
            self.hidden_size2 = hidden_size2
            self.output_size = output_size
            self.linear1 = nn.Linear(self.input_size, self.hidden_size1)
            #self.batchNorm1 = nn.BatchNorm2d(self.hidden_size1)
            self.relu1 = nn.ReLU()
            self.linear2 = nn.Linear(self.hidden_size1, self.hidden_size2)
            #self.batchNorm2 = nn.BatchNorm2d(self.hidden_size2)
            self.relu2 = nn.ReLU()
            self.linear3 = nn.Linear(self.hidden_size2, self.output_size)
            #self.batchNorm2 = nn.BatchNorm2d(self.hidden_size2)
            #self.relu3 = nn.ReLU()
            
        def forward(self, x):
            hidden1 = self.linear1(x)
            relu1  =self.relu1(hidden1)
            hidden2 = self.linear2(relu1)
            relu2 = self.relu2(hidden2)
            output = self.linear3(relu2)
            return output

class WeightedSum(nn.Module):
    def __init__(self):
        super(WeightedSum, self).__init__()
    
    def forward(self, G, input_features):
        G_split_size = 4
        G_ = G[:, :G_split_size]
        L = torch.matmul(torch.transpose(G, 1,0), input_features)
        L_ = torch.matmul(torch.transpose(input_features, 1, 0), G_)
        D = torch.matmul(L, L_)
        return D

class MainNet(nn.Module):
    def __init__(self, input_size1, hidden_size1_1,hidden_size1_2, output_size1,
                 input_size2, hidden_size2_1, hidden_size2_2, output_size2):
        
        super(MainNet, self).__init__()
        self.embeddingNetwork = MLP(input_size1,hidden_size1_1, hidden_size1_2, output_size1)
        self.weightedSum = WeightedSum()
        self.flatten = nn.Flatten(1,-1)
        self.fittingNetwork = MLP(input_size2,hidden_size2_1, hidden_size2_2, output_size2)
        
        
   
    def forward(self, x):
        norm_data = torch.nn.functional.normalize(x)
        #print(f'input values: {x}\n shape: {x.shape}')
        embedding_output = self.embeddingNetwork(norm_data)
        #print(f'embedding output: {embedding_output}\n shape: {embedding_output.shape}')

        #weighted_output = self.weightedSum(embedding_output, x)
        #print(f'weighted output: {weighted_output}\n shape: {weighted_output.shape}')
        
        weighted_output_flattened = torch.flatten(embedding_output)
        #print(f'weighted flattened: {weighted_output_flattened}\n shape: {weighted_output_flattened.shape}')
        
        fitting_output = self.fittingNetwork(weighted_output_flattened)
        #print(f'fitting output: {fitting_output}\n shape: {fitting_output.shape}')

        return fitting_output