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

im_root = '/scratch/additya/graph_data/DB2_A/'
min_root = '/scratch/additya/graph_data/DB2_A_min/'
crop_size = 64
im_list, minX, minY, minQ, labels = [], [], [], [], []
out_im_path = '/scratch/additya/graph_data/DB2_A_cleaned_processed/'
out_min_path = '/scratch/additya/graph_data/DB2_A_cleaned_processed_min/'

def check(fpath, mpath, label):

	if not os.path.exists(mpath):
		return np.array([]), np.array([]), np.array([])

	with open(mpath, 'rb') as fp:
		minu = np.array(pickle.load(fp))

	if minu.shape[0] == 0:
		return np.array([]), np.array([]), np.array([])

	x = np.squeeze(np.array(minu[:, [1]], dtype=np.int))
	y = np.squeeze(np.array(minu[:, [2]], dtype=np.int))
	q = np.squeeze(np.array(minu[:, [-1]], dtype=np.float))

	idx = np.argsort(q)[::-1]
	if idx.shape[0] > 50:
		idx = idx[:50]

	x, y, q = x[idx], y[idx], q[idx]
	return x, y, q

def get_im(im, x, y, l):
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


for root, dirs, files in os.walk(im_root):
	for file in files:
		if file.endswith(('.bmp', '.png')):
			fpath = os.path.join(root, file)
			mpath = fpath.replace(im_root, min_root).replace('.bmp', '.txt')
			mpath = mpath.replace('.bmp', '.txt')
			label = fpath.split('/')[-2]
			x, y, q = check(fpath, mpath, label)
			if x.shape[0] == 0:
				print("Error:", fpath, mpath, label)
			else:
				im_list.append(fpath)
				minX.append(x)
				minY.append(y)
				minQ.append(q)
				labels.append(int(label))
print("Total:", len(im_list))


for idx in range(len(im_list)):

	fpath, label = im_list[idx], labels[idx]
	im = np.array(Image.open(fpath).convert('RGB'))
	x, y, q = minX[idx], minY[idx], minQ[idx]

	cur_min_path = fpath.replace(im_root, out_min_path).replace('.bmp', '.npy')
	cur_minu = np.array([x, y, q])
	cur_min_dir = '/'.join(cur_min_path.split('/')[:-1])
	if not os.path.exists(cur_min_dir):
		os.makedirs(cur_min_dir)
	np.savetxt(cur_min_path, cur_minu)

	cur_im_dir = fpath.replace(im_root, out_im_path).replace('.bmp', '')
	if not os.path.exists(cur_im_dir):
		os.makedirs(cur_im_dir)
	for i in range(len(x)):
		xx, yy = x[i], y[i]
		ret = get_im(im, xx, yy, crop_size)
		spath = os.path.join(cur_im_dir, str(i) + '.png')
		ret = Image.fromarray(ret)
		ret.save(spath)


	if idx % 500 == 0:
		print("Done: ", idx)
