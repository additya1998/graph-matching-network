import os
import torch
import pickle
import numpy as np
from PIL import Image, ImageFile
from torch_geometric.data import Data, Dataset
from torch.utils.data.sampler import BatchSampler

def get_sorted(files):
	ret, t = [], []
	for f in files:
		t.append(int(f.split('.')[0]))
	t = np.array(t);
	ind = np.argsort(t)
	for i in ind:
		ret.append(files[i])
	return ret;


class MinutiaeDataset(Dataset):

	def __init__(self, im_root, min_root, type, input_size=229, \
				 transform=None, pre_transform=None):
		super(MinutiaeDataset, self).__init__()

		print(im_root)

		self.type = type
		self.im_root = im_root
		self.min_root = min_root
		self.input_size = input_size
		self.im_list = {}
		self.ims, self.labels = [], []
		aq_set = set()
		for root, dirs, files in os.walk(self.im_root):
			if self.type == 'test':
				dirs.sort(key=int); files = get_sorted(files);
			for file in files:
				if file.endswith(('.bmp', '.png', '.jpg', '.jpeg')):
					fpath = os.path.join(root, file)
					mpath = fpath.replace(self.im_root, self.min_root)
					mpath = mpath.replace('.png', '.txt')
					mpath = mpath.replace('.bmp', '.txt')
					label = str(fpath.split('/')[-3])
					aq = str(fpath.split('/')[-2])
					if label not in self.im_list.keys():
						self.im_list[label] = {}
					if aq not in self.im_list[label].keys():
						self.im_list[label][aq] = []
					self.im_list[label][aq].append(fpath)
					if (label, aq) not in aq_set:
						self.labels.append(int(label))
						self.ims.append('/'.join(fpath.split('/')[:-1]))
						aq_set.add((label, aq))

		print("Total:", len(self.labels))

	def len(self):
		return len(self.labels)

	def get(self, idx):

		fpath, label = self.ims[idx], self.labels[idx]
		label = str(fpath.split('/')[-2])
		aq = str(fpath.split('/')[-1])

		p_req = min(25, len(self.im_list[label][aq]))

		feat = []
		for p in self.im_list[label][aq]:
			im = Image.open(p).convert('RGB')
			im = im.resize((self.input_size, self.input_size), Image.BICUBIC)
			im = np.array(im)
			im = np.transpose(im, (2, 0, 1))
			im = (im - 127.0) / 127.5
			feat.append(im)
			if len(feat) == p_req:
				break;

		mpath = fpath.replace(self.im_root, self.min_root) + '.npy'
		# print(fpath, mpath)
		mins = np.loadtxt(mpath)
		(x, y) = (mins[0, :], mins[1, :])

		source_nodes, target_nodes = [], []
		for i in range(len(x)):
			for j in range(len(x)):
					if i >= p_req or j >= p_req:
						break;
					if i == j:
						continue
					xa, ya = x[i], y[j]
					xb, yb = x[j], y[j]
					d = ((xa - xb) * (xa - xb)) + ((ya - yb) * (ya - yb))
					if d < 100:
						source_nodes.append(i)
						target_nodes.append(j)

		edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
		feat = torch.FloatTensor(feat)

		return Data(x=feat, edge_index=edge_index), int(label)

class BalancedBatchSampler(BatchSampler):

	def __init__(self, labels, n_classes, n_samples):

		self.labels = np.array(labels)
		self.labels_set = list(set(self.labels))
		self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}
		for l in self.labels_set:
			np.random.shuffle(self.label_to_indices[l])
		self.used_label_indices_count = {label: 0 for label in self.labels_set}
		self.count = 0
		self.n_classes = n_classes
		self.n_samples = n_samples
		self.n_dataset = self.labels.shape[0]
		self.batch_size = self.n_samples * self.n_classes

	def __iter__(self):
		self.count = 0

		while self.count + self.batch_size < self.n_dataset:
			classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
			indices = []
			for class_ in classes:
				indices.extend(self.label_to_indices[class_][
							   self.used_label_indices_count[class_]:self.used_label_indices_count[
																		 class_] + self.n_samples])
				self.used_label_indices_count[class_] += self.n_samples
				if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
					np.random.shuffle(self.label_to_indices[class_])
					self.used_label_indices_count[class_] = 0
			yield indices
			self.count += self.n_classes * self.n_samples

	def __len__(self):
		return (self.n_dataset) // self.batch_size
