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

# remove id column from target data
data_target = data_target.drop(columns=['id'])

# rename new_id column to id
data_target = data_target.rename(columns={'new_id': 'id'})

# rename from to from_id and to to to_id
data_edges = data_edges.rename(columns={'from': 'from_id', 'to': 'to_id'})

# Change mature and partner boolean to integer
data_target['mature'] = data_target['mature'].astype(int)
data_target['partner'] = data_target['partner'].astype(int)

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
