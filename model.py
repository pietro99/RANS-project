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
        self.G_split_size = 5
    
    def forward(self, G, input_features):
        #print(f'if: {input_features[0]}')
        G_ = G[:, :self.G_split_size]
        #print(G_.shape[0])
        L = (torch.matmul(torch.transpose(G, 1,0), input_features) / G.shape[0])
        L_ = (torch.matmul(torch.transpose(input_features, 1, 0), G_) / G.shape[0])
        D = torch.matmul(L, L_)
        #print(f'D: {D}')
        #print()
        return D
    
class MainNet(nn.Module):
    def __init__(self, input_size1, hidden_size1_1,hidden_size1_2, output_size1,
                 input_size2, hidden_size2_1, hidden_size2_2, output_size2):
        
        super(MainNet, self).__init__()
        self.invariant_features_indeces = [4,5,6,7]

        self.embeddingNetwork = MLP(input_size1,hidden_size1_1, hidden_size1_2, output_size1)
        self.weightedSum = WeightedSum()
        self.flatten = nn.Flatten(1,-1)
        self.fittingNetwork = MLP(input_size2,hidden_size2_1, hidden_size2_2, output_size2)
        
        
   
    def forward(self, x):
        #print(f'input values: {x[0]}\n shape: {x.shape}')
        invariant_x = x[:, self.invariant_features_indeces]
        
        embedding_output = self.embeddingNetwork(invariant_x)
        #print(f' shape: {embedding_output.shape}')

        weighted_output = self.weightedSum(embedding_output, x)
        #print(f' shape: {weighted_output.shape}')
        
        weighted_output_flattened = torch.flatten(weighted_output)
        #print(f' shape: {weighted_output_flattened.shape}')
        
        fitting_output = self.fittingNetwork(weighted_output_flattened)
        #print(f'fitting output: {fitting_output}\n shape: {fitting_output.shape}')

        return fitting_output