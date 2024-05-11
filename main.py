import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch.optim as optim


# Load the data
data_edges = pd.read_csv('DE_edges.csv')
data_target = pd.read_csv('DE_target.csv')

# remove all column besides id and mature from target data
data_target = data_target.drop(columns=['id'])
data_target = data_target.drop(columns=['partner'])
data_target = data_target.drop(columns=['views'])
data_target = data_target.drop(columns=['days'])

# rename new_id column to id
data_target = data_target.rename(columns={'new_id': 'id'})

# rename from to from_id and to to to_id
data_edges = data_edges.rename(columns={'from': 'from_id', 'to': 'to_id'})

# Change mature and partner boolean to integer
data_target['mature'] = data_target['mature'].astype(int)

# Print the data
print("Edges data:" )
print(data_edges.head())
print("Target data:" )
print(data_target.head())

# print data statistics
#print("Edges data statistics:")
#print(data_edges.describe())
#print("Target data statistics:")
#print(data_target.describe())

# Prepare edge index and mapping to target
edge_index = torch.tensor(data_edges.values, dtype=torch.long).t().contiguous()
node_ids = data_target['id']
node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
labels = torch.tensor(data_target['mature'].values, dtype=torch.long)

# Prepare PyG data object
x = torch.eye(len(node_ids))  # simplified node feature initialization: identity matrix
train_indices, test_indices = train_test_split(range(len(node_ids)), test_size=0.2, stratify=labels)
train_mask = torch.zeros(len(node_ids), dtype=torch.bool).scatter_(0, torch.tensor(train_indices), True)
test_mask = torch.zeros(len(node_ids), dtype=torch.bool).scatter_(0, torch.tensor(test_indices), True)

data = Data(x=x, edge_index=edge_index, y=labels, train_mask=train_mask, test_mask=test_mask)

# Define GCN model
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(data.num_features, 32)
        self.conv2 = GCNConv(32, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN().to(device)
data = data.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
    return correct / int(data.test_mask.sum())

# Training loop
epochs = 200
for epoch in range(epochs):
    loss = train()
    acc = test()
    print(f'Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')