import pandas as pd
import torch
import json
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
with open('DE.json') as f:
    node_features_json = json.load(f)

# Convert the JSON data to a DataFrame
node_features_df = pd.DataFrame.from_dict(node_features_json, orient='index')

# Rename the index to 'id' to match the other data
node_features_df.index.name = 'id'

# Reset the index so 'id' becomes a column
node_features_df.reset_index(inplace=True)

# remove all column besides id and mature from target data
data_target = data_target.drop(columns=['id'])
#data_target = data_target.drop(columns=['partner'])
#data_target = data_target.drop(columns=['views'])
#data_target = data_target.drop(columns=['days'])

# rename new_id column to id
data_target = data_target.rename(columns={'new_id': 'id'})

# rename from to from_id and to to to_id
data_edges = data_edges.rename(columns={'from': 'from_id', 'to': 'to_id'})

# Change mature and partner boolean to integer
data_target['mature'] = data_target['mature'].astype(int)
data_target['partner'] = data_target['partner'].astype(int)

#cyk normalizacja
data_target['days'] = (data_target['days'] - data_target['days'].mean()) / data_target['days'].std()
data_target['views'] = (data_target['views'] - data_target['views'].mean()) / data_target['views'].std()

# Prepare PyG data object
node_features = torch.tensor(data_target.drop(columns=['id']).values, dtype=torch.float)  # use all columns except 'id' as node features
data_target['id'] = data_target['id'].astype('int64')
node_features_df['id'] = node_features_df['id'].astype('int64')

# Now merge
data_target = pd.merge(data_target, node_features_df, on='id')

#place 0 insted of NaN
data_target = data_target.fillna(0)
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
#make edge_index undirected
edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

node_ids = data_target['id']
node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
labels = torch.tensor(data_target['mature'].values, dtype=torch.long)

# Prepare PyG data object
x = torch.eye(len(node_ids))  # simplified node feature initialization: identity matrix
train_indices, test_indices = train_test_split(range(len(node_ids)), test_size=0.20, stratify=labels)
train_mask = torch.zeros(len(node_ids), dtype=torch.bool).scatter_(0, torch.tensor(train_indices), True)
test_mask = torch.zeros(len(node_ids), dtype=torch.bool).scatter_(0, torch.tensor(test_indices), True)

data = Data(x=node_features, edge_index=edge_index, y=labels, train_mask=train_mask, test_mask=test_mask)

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(node_features.shape[1], 32)
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
#other loss functions
#criterion = nn.CrossEntropyLoss()

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
    accuracy = correct / int(data.test_mask.sum())

    return accuracy

# Training loop
epochs = 2000
for epoch in range(epochs):
    loss = train()
    acc = test()
    print(f'Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')