import pandas as pd
import torch
import json
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch.optim as optim
import matplotlib.pyplot as plt

SOURCE_PATH = 'dataset/'

def load_data():
    data_edges = pd.read_csv(SOURCE_PATH + 'DE_edges.csv') # Load the data from the csv files
    data_target = pd.read_csv(SOURCE_PATH + 'DE_target.csv') 

    with open(SOURCE_PATH + 'DE.json') as f: # Load the node features from the json file
        node_features_json = json.load(f)
    
    node_features_df = pd.DataFrame.from_dict(node_features_json, orient='index') # Convert the json to a pandas dataframe
    node_features_df.index.name = 'id' 
    node_features_df.reset_index(inplace=True)

    return data_edges, data_target, node_features_df

def preprocess_data(data_target, data_edges, node_features_df):

    data_target = data_target.drop(columns=['id']) # Drop the id column (not needed for the model)
    data_target = data_target.rename(columns={'new_id': 'id'}) # Rename the new_id column to id (because id is deprecated)
    data_edges = data_edges.rename(columns={'from': 'from_id', 'to': 'to_id'}) # Rename the columns to match the node features

    data_target['mature'] = data_target['mature'].astype(int) # Convert the mature and partner column to int
    data_target['partner'] = data_target['partner'].astype(int)

    data_target['days'] = (data_target['days'] - data_target['days'].min()) / (data_target['days'].max() - data_target['days'].min()) # Normalize data
    data_target['views'] = (data_target['views'] - data_target['views'].min()) / (data_target['views'].max() - data_target['views'].min())

    node_features = torch.tensor(data_target.drop(columns=['id']).values, dtype=torch.float) # Convert the node features to a tensor

    data_target['id'] = data_target['id'].astype('int64') # Convert the column to int64
    node_features_df['id'] = node_features_df['id'].astype('int64')
    data_target = pd.merge(data_target, node_features_df, on='id')

    data_target = data_target.fillna(0) # Repleace NaN values with 0

    return data_target, data_edges, node_features

def prepare_data(data_target, data_edges, node_features):
    edge_index = torch.tensor(data_edges.values, dtype=torch.long).t().contiguous() # Convert the edge list to a tensor
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1) # Add the reverse edge list

    node_ids = data_target['id'] # Get the node ids
    labels = torch.tensor(data_target['mature'].values, dtype=torch.long) # Get the labels

    train_indices, test_indices = train_test_split(range(len(node_ids)), test_size=0.20, stratify=labels) # Split the data into train and test sets
    train_mask = torch.zeros(len(node_ids), dtype=torch.bool).scatter_(0, torch.tensor(train_indices), True) # Create the train and test masks
    test_mask = torch.zeros(len(node_ids), dtype=torch.bool).scatter_(0, torch.tensor(test_indices), True) 

    data = Data(x=node_features, edge_index=edge_index, y=labels, train_mask=train_mask, test_mask=test_mask) # Create the data object

    return data

class GCN(nn.Module):
    def __init__(self, node_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(node_features.shape[1], 32)
        self.conv2 = GCNConv(32, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
    accuracy = correct / int(data.test_mask.sum())
    return accuracy

def plot_results(loss_values, accuracy_values):
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(loss_values, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over time')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_values, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over time')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    data_edges, data_target, node_features_df = load_data()
    data_target, data_edges, node_features = preprocess_data(data_target, data_edges, node_features_df)
    data = prepare_data(data_target, data_edges, node_features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN(node_features).to(device)
    data = data.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss() 
    epochs = 2000
    loss_values = []
    accuracy_values = []
    for epoch in range(epochs):
        loss = train(model, data, optimizer, criterion)
        acc = test(model, data)
        loss_values.append(loss)
        accuracy_values.append(acc)
        print(f'Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

    plot_results(loss_values, accuracy_values)

if __name__ == "__main__":
    main()