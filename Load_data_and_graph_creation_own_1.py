import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv(r'C:\Users\hi tech\PycharmProjects\Test_projects\Wind_onshoreMW.csv', parse_dates=['Datetime'],
                 index_col='Datetime')


# Convert the time series data into a graph
def create_graph(data):
    x = torch.tensor(data.values, dtype=torch.float).unsqueeze(1)
    edge_index = torch.tensor(
        [[i, i + 1] for i in range(len(data) - 1)] + [[i + 1, i] for i in range(len(data) - 1)], dtype=torch.long
    ).t().contiguous()
    return Data(x=x, edge_index=edge_index)


graph_data = create_graph(df['Wind Onshore_MW'])


# Define the GCN model
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


# Training loop with loss tracking
losses = []
for epoch in range(200):
    loss = train()
    losses.append(loss)
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Evaluate the model
def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        predictions = model(data).numpy()
    true_values = data.x.numpy()

    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predictions - true_values))

    # Calculate Mean Squared Error (MSE)
    mse = np.mean((predictions - true_values) ** 2)

    return mae, mse


mae, mse = evaluate(model, graph_data)
print(f'Mean Absolute Error: {mae:.4f}')
print(f'Mean Squared Error: {mse:.4f}')

# Visualize the results
model.eval()
with torch.no_grad():
    predictions = model(graph_data).numpy()

plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Wind Onshore_MW'], label='Original')
plt.plot(df.index, predictions, label='Predicted', linestyle='--')
plt.xlabel('Datetime')
plt.ylabel('Wind Onshore MW')
plt.legend()
plt.show()
