import os
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms


class DataLoader(object):
	"""
	data loader for CV data sets
	"""
	
	def __init__(self, dataset, batch_size):
		"""
		create data loader for specific data set
		:params n_treads: number of threads to load data, default: 4
		:params data_path_root: root path to data set, default: /lustre/datasets/
		"""
		self.dataset = dataset
		self.batch_size = batch_size
		self.n_threads = 4 #num_workers
		self.data_path_root = '/lustre/datasets/'
		
		
		if self.dataset in ["cifar100","cifar10"]:
			self.train_loader, self.val_loader, self.test_loader = self.cifar(
				dataset=self.dataset)
		else:
			assert False, "invalid data set"
	
	def getloader(self):
		"""d
		get train_loader and test_loader
		"""
		return self.train_loader, self.val_loader, self.test_loader

	def cifar(self, dataset):
		"""
		dataset: cifar
		"""
		if dataset == "cifar10":
			norm_mean = [0.49139968, 0.48215827, 0.44653124]
			norm_std = [0.24703233, 0.24348505, 0.26158768]
		elif dataset == "cifar100":
			norm_mean = [0.50705882, 0.48666667, 0.44078431]
			norm_std = [0.26745098, 0.25568627, 0.27607843]
		else:
			assert False, "Invalid cifar dataset"

		train_transfrom = transforms.Compose([
							transforms.RandomCrop(32, padding=2),
							transforms.RandomHorizontalFlip(),
							transforms.ToTensor(),
							transforms.Normalize(norm_mean, norm_std)
							])
		eval_transfrom = transforms.Compose([
							transforms.ToTensor(),
							transforms.Normalize(norm_mean, norm_std)
							])

		if self.dataset == "cifar10":
			data_path = self.data_path_root + 'CIFAR10'
			alltrainset = dsets.CIFAR10(root=data_path,train=True,download=False,
										transform=train_transfrom)
			testset = dsets.CIFAR10(data_path, train=False, download=False, 
									transform=eval_transfrom)
		elif self.dataset == "cifar100":
			data_path = self.data_path_root + 'CIFAR100'
			alltrainset = dsets.CIFAR100(root=data_path,train=True,download=False,
										transform=train_transfrom)
			testset = dsets.CIFAR100(data_path, train=False, download=False, 
									transform=eval_transfrom)
		else:
			assert False, "invalid data set"

		train_size = (int)(0.8 * len(alltrainset))
		val_size = (int)(0.2 * len(alltrainset))
		
		train_idx, val_idx = torch.utils.data.random_split(range(train_size+val_size),[train_size,val_size])

		trainset = torch.utils.data.Subset(alltrainset,train_idx)
		valset = torch.utils.data.Subset(alltrainset,val_idx)

		train_loader = torch.utils.data.DataLoader(
		trainset,
		batch_size=self.batch_size, shuffle=True, num_workers=self.n_threads, pin_memory=True
		)

		val_loader = torch.utils.data.DataLoader(
			valset,
			batch_size=self.batch_size, shuffle=False, num_workers=self.n_threads, pin_memory=True
		)

		test_loader = torch.utils.data.DataLoader(
			testset,
			batch_size=self.batch_size, shuffle=False, num_workers=self.n_threads, pin_memory=True
		)

		return train_loader, val_loader, test_loader
	

