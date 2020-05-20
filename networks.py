import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv, SAGEConv, SGConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

class NodeEncoder(nn.Module):

	def __init__(self,  embedding_size=128):
		super(NodeEncoder, self).__init__()

		self.embedding_size = embedding_size

		self.conv_layers = []
		self.conv_layers.append(BasicConv2d(1, 32, kernel_size=3, stride=2).cuda())
		self.conv_layers.append(BasicConv2d(32, 32, kernel_size=3, stride=2).cuda())
		self.conv_layers.append(BasicConv2d(32, 64, kernel_size=3, stride=2).cuda())
		self.conv_layers.append(BasicConv2d(64, 128, kernel_size=3, stride=2).cuda())
		self.fc = nn.Linear(128, embedding_size)

	def forward(self, data):
		x = torch.unsqueeze(data.x, 1)
		for conv_layer in self.conv_layers:
			x = conv_layer(x)
		x = F.adaptive_avg_pool2d(x, (1, 1))
		x = torch.flatten(x, 1)
		x = self.fc(x)
		data.x = x
		return data

class GraphPropagator(nn.Module):

	def __init__(self):
		super(GraphPropagator, self).__init__()

		# self.node_encoder = node_encoder

		self.conv1 = SAGEConv(512, 512)
		self.pool1 = TopKPooling(512, ratio=0.8)
		self.conv2 = SAGEConv(512, 512)
		self.pool2 = TopKPooling(512, ratio=0.8)
		self.conv3 = SAGEConv(512, 512)
		self.pool3 = TopKPooling(512, ratio=0.8)

		self.lin1 = torch.nn.Linear(1024, 512)
		self.lin2 = torch.nn.Linear(512, 256)
	
	def forward(self, data):
		x, edge_index, batch = data.x, data.edge_index, data.batch

		x = F.relu(self.conv1(x, edge_index))
		x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
		x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

		x = F.relu(self.conv2(x, edge_index))
		x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
		x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

		x = F.relu(self.conv3(x, edge_index))
		x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
		x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

		x = x1 + x2 + x3

		x = F.relu(self.lin1(x))
		x = F.dropout(x, p=0.5, training=self.training)
		x = self.lin2(x)

		return x

class GraphEncoder(nn.Module):
	
	def __init__(self):
		super(GraphEncoder, self).__init__()

	def forward(self, x):
		return global_add_pool(x.x, x.batch)

class Net(nn.Module):

	def __init__(self, node_encoder, graph_propagator, graph_encoder):
		super(Net, self).__init__()

		self.node_encoder = node_encoder
		self.graph_propagator = graph_propagator
		self.graph_encoder = graph_encoder

	def forward(self, x):
		# print('inp', x)
		# print(x.x.shape)
		# print(x.x.shape, torch.unsqueeze(x.x, 1).shape)
		x.x = self.node_encoder(x.x)
		# print(x)
		# print('node enc', x, x.x)
		x = self.graph_propagator(x)
		# print('graph prop', x)
		# x = self.graph_encoder(x)
		# print(x)
		return x

class BasicConv1d(nn.Module):

	def __init__(self, in_channels, out_channels, **kwargs):
		super(BasicConv1d, self).__init__()
		self.conv = nn.Conv1d(in_channels, out_channels, bias=False, **kwargs)
		self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		return F.relu(x, inplace=True)

class BasicConv2d(nn.Module):

	def __init__(self, in_channels, out_channels, **kwargs):
		super(BasicConv2d, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
		self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		return F.relu(x, inplace=True)

class SAGEConv(MessagePassing):
	def __init__(self, in_channels, out_channels):
		super(SAGEConv, self).__init__(aggr='add')
		self.lin = torch.nn.Linear(in_channels, out_channels)
		self.act = torch.nn.ReLU()
		self.update_lin = torch.nn.Linear(in_channels + out_channels, in_channels, bias=False)
		self.update_act = torch.nn.ReLU()
		
	def forward(self, x, edge_index):
		# x has shape [N, in_channels]
		# edge_index has shape [2, E]
				
		edge_index, _ = remove_self_loops(edge_index)
		edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
		
		return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

	def message(self, x_j):
		# x_j has shape [E, in_channels]

		x_j = self.lin(x_j)
		x_j = self.act(x_j)
		
		return x_j

	def update(self, aggr_out, x):
		# aggr_out has shape [N, out_channels]

		new_embedding = torch.cat([aggr_out, x], dim=1)
		new_embedding = self.update_lin(new_embedding)
		new_embedding = self.update_act(new_embedding)
		
		return new_embedding

