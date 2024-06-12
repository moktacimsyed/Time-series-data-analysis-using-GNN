#####################
# Preprocess the data
#####################

import pandas as pd
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt

# Load the dataset from the CSV file
df = pd.read_csv(r'C:\Users\hi tech\PycharmProjects\Test_projects\Wind_onshoreMW.csv', parse_dates=['Datetime'], index_col='Datetime')

# Display the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Fill missing values (you can choose a different method  based on your data)
# data = df.fillna(method='ffill')

#################33
# Create graph structure
# To create a graph structure, we need to represent each time point as a node and create edges between consecutive time points.
######################

# Convert the time series data into a graph
def create_graph(data):
    x = torch.tensor(data.values, dtype=torch.float).unsqueeze(1)  # Ensure shape is [num_nodes, num_features]
    edge_index = torch.tensor(
        [[i, i+1] for i in range(len(data)-1)] + [[i+1, i] for i in range(len(data)-1)], dtype=torch.long  # Bidirectional edges
    ).t().contiguous()
    return Data(x=x, edge_index=edge_index)

# Create the graph
graph_data = create_graph(df['Wind Onshore_MW'])
print(graph_data)


#####################
# Build and Train GNN model
##########################


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Define the model
model = GCN(in_channels=1, hidden_channels=16, out_channels=1)

# Training the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(graph_data)
    loss = criterion(out, graph_data.x)
    loss.backward()
    optimizer.step()
    return loss.item()

# Training loop
for epoch in range(200):
    loss = train()
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

###################
# Visualize the result
#################

# Get the model predictions
model.eval()
with torch.no_grad():
    predictions = model(graph_data).numpy()

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Wind Onshore_MW'], label='Original')
plt.plot(df.index, predictions, label='Predicted', linestyle='--')
plt.xlabel('Datetime')
plt.ylabel('Wind Onshore MW')
plt.legend()
plt.show()
