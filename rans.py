import torch
import torch.nn as nn



class MLP(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
            super(MLP, self).__init__()
            self.input_size = input_size
            self.hidden_size1  = hidden_size1
            self.hidden_size2 = hidden_size2
            self.output_size = output_size
            self.linear1 = nn.Linear(self.input_size, self.hidden_size1)
            self.relu1 = nn.ReLU()
            self.linear2 = nn.Linear(self.hidden_size1, self.hidden_size2)
            self.relu2 = nn.ReLU()
            self.linear3 = nn.Linear(self.hidden_size2, self.output_size)
            self.relu3 = nn.ReLU()
            
        def forward(self, x):
            hidden1 = self.linear1(x)
            relu1  =self.relu1(hidden1)
            hidden2 = self.linear2(relu1)
            relu2 = self.relu2(hidden2)
            hidden3 = self.linear3(relu2)
            output = self.relu3(hidden3)
            return output

stencil_size = 20
input_features = torch.rand(stencil_size, 11)
rotation_inv_features =input_features[:,4:]
y = torch.rand(stencil_size, 2)

input_size = rotation_inv_features.size()[1]
hidden_size1 = 32
hidden_size2 = 64
output_size = 64
embeddingNetwork = MLP(input_size,hidden_size1, hidden_size2, output_size)
loss = torch.nn.BCELoss()
optimizer = torch.optim.SGD(embeddingNetwork.parameters(), lr = 0.01)

G = embeddingNetwork(rotation_inv_features)

G_split_size = 4
G_ = G[:, :G_split_size]

L = torch.matmul(torch.transpose(G, 1,0), input_features)
L_ = torch.matmul(torch.transpose(input_features, 1, 0), G_)

D = torch.matmul(L, L_)

D_flattened = D.flatten()

input_size = D_flattened.size()[0]
hidden_size1 = 128
hidden_size2 = 64
output_size = 2
fittingNetwork = MLP(input_size, hidden_size1, hidden_size2, output_size)

optimizer = torch.optim.SGD(fittingNetwork.parameters(), lr = 0.01)

res = fittingNetwork(D_flattened)

k, epsilon = res.detach().numpy()
print(f'k       : {k:.4f}\nepsilon : {epsilon:.4f}')
