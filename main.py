import os
import sys
import numpy as np

import torch
import torch.optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torchvision import models

from metrics import get_auth_metrics_mcc
from losses import OnlineTripletLoss
from utils import SemihardNegativeTripletSelector
from networks import Net, GraphEncoder, GraphPropagator, NodeEncoder
from dataset import MinutiaeDataset, BalancedBatchSampler
from resnet import resnet18, resnet50


im_dir = '/home/additya/fp_data/nist_sd_302_cleaned_sample_processed/'
min_dir = '/home/additya/fp_data/nist_sd_302_cleaned_sample_processed_min/'
# test_im_dir = '/home/additya/delete/fvc_db2a/'
# test_min_dir = '/home/additya/delete/fvc_db2a_min/'

im_dir = '/scratch/additya/graph_data/nist_sd_302_cleaned_processed'
min_dir = '/scratch/additya/graph_data/nist_sd_302_cleaned_processed_min'
test_im_dir = '/scratch/additya/graph_data/DB2_A_cleaned_processed'
test_min_dir = '/scratch/additya/graph_data/DB2_A_cleaned_processed_min'
# # im_dir = test_im_dir
# # min_dir = test_min_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
device_ids = [int(device_id) for device_id in device_ids]
print('GPU configuration:', device, device_ids)

# # device = torch.device('cuda')
# # encoder = NodeEncoder().to(device)
encoder = resnet18(pretrained=True)
encoder.fc = torch.nn.Linear(512, 512)
propagator = GraphPropagator().to(device)
gencoder = GraphEncoder().to(device)
model = Net(encoder, propagator, gencoder)
# model = torch.nn.DataParallel(model, device_ids=device_ids).to(device)
# clf = torch.nn.Linear(256, 2000)
# clf = torch.nn.DataParallel(clf, device_ids=device_ids).to(device)
model = model.to(device)
clf = torch.nn.Linear(256, 2000).to(device)


criterion = torch.nn.CrossEntropyLoss()
# # criterion = OnlineTripletLoss(2.0, SemihardNegativeTripletSelector(2.0))

# print("Net done!")

train_dataset = MinutiaeDataset(im_dir, min_dir, 'train')
# # sampler = BalancedBatchSampler(dataset.labels, n_classes=10, n_samples=5)
# # train_loader = DataLoader(dataset, num_workers=10, batch_sampler=sampler)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=10)

test_dataset = MinutiaeDataset(test_im_dir, test_min_dir, 'test')
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=10)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)

for epoch in range(10):
	print("Epoch:", epoch); print('#' * 25)
	sys.stdout.flush()

	model.train()
	total_loss, total_samples = 0, 0
	for (idx, (data, labels)) in enumerate(train_loader):
		labels = labels.type(torch.LongTensor)
		(data, labels) = (data.to(device), labels.to(device))
		optimizer.step()
		optimizer.zero_grad()
		with torch.set_grad_enabled(True):
			outputs = model(data)
			probabilities = clf(outputs)
			loss = criterion(probabilities, labels)
			loss.backward()
			total_samples =  total_samples + labels.shape[0]
			total_loss = total_loss + (loss.item() * labels.shape[0])
			if idx % 100 == 0:
				print("Step " + str(idx) + ":", loss.item())
			sys.stdout.flush()
		scheduler.step()
	print("Epoch Loss:", np.round(total_loss / total_samples, 4))
'''
	model.eval()
	(total_e, total_labels) = (np.array([]), np.array([]))
	for (data, labels) in test_loader:
		labels = labels.type(torch.LongTensor)
		(data, labels) = (data.to(device), labels.to(device))
		with torch.set_grad_enabled(False):
			outputs = model(data)
			# print(outputs.shape)
			outputs = F.normalize(outputs, p=2, dim=1)
			if total_e.shape[0]:
				total_e = np.vstack((total_e, outputs.cpu().numpy()))
				total_labels = np.hstack((total_labels, labels.cpu().numpy()))
			else:
				total_e = outputs.cpu().numpy()
				total_labels = labels.cpu().numpy()
	print('Embeddings:', total_e.shape)
	get_auth_metrics_mcc(total_e, total_labels, 'test')

	torch.save(model.state_dict(), '/scratch/additya/' + str(epoch) + '.pth')
'''
