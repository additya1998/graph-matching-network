import os
import cv2
import torch
import pickle
import numpy as np
from PIL import Image, ImageFile
from torch_geometric.data import Data, Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms

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
	
	def __init__(self, im_root, min_root, type, crop_size=64, input_size=299, \
				 transform=None, pre_transform=None):
		super(MinutiaeDataset, self).__init__()

		self.type = type
		self.im_root = im_root
		self.min_root = min_root
		self.crop_size = crop_size
		self.input_size = input_size
		self.im_list, self.minX, self.minY, self.minQ, self.labels = [], [], [], [], []
		for root, dirs, files in os.walk(self.im_root):
			if self.type == 'test':
				dirs.sort(key=int); files = get_sorted(files);
			for file in files:
				if file.endswith(('.bmp', '.png')):
					fpath = os.path.join(root, file)
					mpath = fpath.replace(self.im_root, self.min_root).replace('.png', '.txt')
					mpath = mpath.replace('.bmp', '.txt')
					label = fpath.split('/')[-2]
					x, y, q = self.check(fpath, mpath, label)
					if x.shape[0] == 0:
						print("Error:", fpath, mpath, label)
					else:
						self.im_list.append(fpath)
						self.minX.append(x)
						self.minY.append(y)
						self.minQ.append(q)
						self.labels.append(int(label))
		print("Total:", len(self.im_list))

	def check(self, fpath, mpath, label):
		with open(mpath, 'rb') as fp:
			minu = np.array(pickle.load(fp))
		
		if minu.shape[0] == 0:
			return np.array([]), np.array([]), np.array([])

		x = np.squeeze(np.array(minu[:, [1]], dtype=np.int)) 
		y = np.squeeze(np.array(minu[:, [2]], dtype=np.int))
		q = np.squeeze(np.array(minu[:, [-1]], dtype=np.float))

		idx = np.argsort(q)[::-1]
		if idx.shape[0] > 30:
			idx = idx[:30]

		x, y, q = x[idx], y[idx], q[idx]
		return x, y, q

	def len(self):
		return len(self.im_list)

	def get_im(self, im, x, y, l):
		minx, maxx = x - l // 2, x + l // 2
		miny, maxy = y - l // 2, y + l // 2

		minx, miny = x - l // 2, y - l // 2
		maxx, maxy = x + l // 2, y + l // 2

		if minx < 0:
			x = x - minx
			minx, maxx = x - l // 2, x + l // 2
		if miny < 0:
			y = y - miny
			miny, maxy = y - l // 2, y + l // 2
		if maxx > im.shape[0]:
			x = x + (im.shape[0] - maxx)
			minx, maxx = x - l // 2, x + l // 2
		if maxy > im.shape[1]:
			y = y + (im.shape[1] - maxy)
			miny, maxy = y - l // 2, y + l // 2

		if len(im.shape) == 2:
			return im[minx:maxx, miny:maxy]
		else:
			return im[minx:maxx, miny:maxy, :]

	def get(self, idx):

		fpath, label = self.im_list[idx], self.labels[idx]
		im = np.array(Image.open(fpath).convert('RGB'))
		x, y, q = self.minX[idx], self.minY[idx], self.minQ[idx]

		feat = []
		for i in range(len(x)):
			xx, yy = x[i], y[i]
			ret = self.get_im(im, xx, yy, self.crop_size)
			ret = np.transpose(ret, (2, 0, 1))
			# print(ret)
			# ret = Image.fromarray(ret)
			# ret = np.array(ret.resize((self.input_size, self.input_size)))
			ret = (ret - 127.0) / 128.0
			ret = cv2.resize(ret, (self.input_size, self.input_size), cv2.INTER_CUBIC)
			feat.append(ret)

		source_nodes, target_nodes = [], []
		for i in range(len(x)):
			for j in range(len(x)):
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

		return Data(x=feat, edge_index=edge_index), label

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
