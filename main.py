import pandas as pd
import torch
import json
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch.optim as optim

def load_data():
    data_edges = pd.read_csv('DE_edges.csv')
    data_target = pd.read_csv('DE_target.csv')
    with open('DE.json') as f:
        node_features_json = json.load(f)
    node_features_df = pd.DataFrame.from_dict(node_features_json, orient='index')
    node_features_df.index.name = 'id'
    node_features_df.reset_index(inplace=True)
    return data_edges, data_target, node_features_df

def preprocess_data(data_target, data_edges, node_features_df):
    data_target = data_target.drop(columns=['id'])
    data_target = data_target.rename(columns={'new_id': 'id'})
    data_edges = data_edges.rename(columns={'from': 'from_id', 'to': 'to_id'})
    data_target['mature'] = data_target['mature'].astype(int)
    data_target['partner'] = data_target['partner'].astype(int)
    data_target['days'] = (data_target['days'] - data_target['days'].mean()) / data_target['days'].std()
    data_target['views'] = (data_target['views'] - data_target['views'].mean()) / data_target['views'].std()
    node_features = torch.tensor(data_target.drop(columns=['id']).values, dtype=torch.float)
    data_target['id'] = data_target['id'].astype('int64')
    node_features_df['id'] = node_features_df['id'].astype('int64')
    data_target = pd.merge(data_target, node_features_df, on='id')
    data_target = data_target.fillna(0)
    return data_target, data_edges, node_features

def prepare_data(data_target, data_edges, node_features):
    edge_index = torch.tensor(data_edges.values, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    node_ids = data_target['id']
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    labels = torch.tensor(data_target['mature'].values, dtype=torch.long)
    x = torch.eye(len(node_ids))
    train_indices, test_indices = train_test_split(range(len(node_ids)), test_size=0.20, stratify=labels)
    train_mask = torch.zeros(len(node_ids), dtype=torch.bool).scatter_(0, torch.tensor(train_indices), True)
    test_mask = torch.zeros(len(node_ids), dtype=torch.bool).scatter_(0, torch.tensor(test_indices), True)
    data = Data(x=node_features, edge_index=edge_index, y=labels, train_mask=train_mask, test_mask=test_mask)
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

def main():
    data_edges, data_target, node_features_df = load_data()
    data_target, data_edges, node_features = preprocess_data(data_target, data_edges, node_features_df)
    data = prepare_data(data_target, data_edges, node_features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN(node_features).to(device)
    data = data.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    #criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss() # +1% accuracy
    epochs = 2000
    for epoch in range(epochs):
        loss = train(model, data, optimizer, criterion)
        acc = test(model, data)
        print(f'Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

        import matplotlib.pyplot as plt

        def draw_chart(loss_values, acc_values):
            plt.plot(loss_values, label='Loss')
            plt.plot(acc_values, label='Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend()
            plt.show()

        if __name__ == "__main__":
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
            acc_values = []
            for epoch in range(epochs):
                loss = train(model, data, optimizer, criterion)
                acc = test(model, data)
                loss_values.append(loss)
                acc_values.append(acc)
                print(f'Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
            draw_chart(loss_values, acc_values)

if __name__ == "__main__":
    main()