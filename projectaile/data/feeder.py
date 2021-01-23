import os
import librosa
import numpy as np
import pandas as pd

from .loaders import loaders
from .preprocesses import PREPROCESSES
from .augmentations import AUGMENTATIONS
from .data_utils.feeder_utils import *

'''
	FEEDER : FEEDER class for getting batches from the loader
			and feeding them to the model trainer
'''
class FEEDER:
	def __init__(self, config, loader=None):
		self.config = config
		self.initialize(loader)
	
	def initialize(self, loader):
		self.get_dset_info()
		if not loader:
			loader = self.get_loader(self.config.DATA.DATA_TYPE)
		
		self.loader = loader(self.config)
		self.preprocessor = PREPROCESSES(self.config.DATA.PREPROCESSES)
		self.augmentor = AUGMENTATIONS(self.config.DATA.AUGMENTATIONS)
		self.train_iterator = 0
		self.valid_iterator = 0

	'''
	get_loader : returns one of default loaders based on the input type in config
	'''
	def get_loader(self, dtype):
		if dtype in loaders.keys():
			return loaders[dtype]
		else:
			print('No Default Loader Found For Given Data Type! Please Implement A Custom Data Loader Or Look At The Existing Loaders.')
			return None
		
	'''
		get next set of indices for the loader

		iterator : the iterator index for either train or valid batch (step)
		indices : the indices of train or valid data
		features : the features for train or valid data
		targets : the targets for train or valid data
		batch_size : the number of samples to load
	'''
	def next(self, iterator, indices, features, targets, batch_size):
		batch_indices = indices[iterator*batch_size:(iterator+1)*batch_size]
		iterator += 1

		features = features[batch_indices]
		targets = targets[batch_indices]

		return features, targets, iterator

	'''
		get_batch : get the next batch of data and apply preprocessing and augmentations steps
		iterator : the iterator index for either train or valid batch (step)
		indices : the indices of train or valid data
		features : the features for train or valid data
		targets : the targets for train or valid data
		batch_size : the number of samples to load
	'''
	def get_batch(self, iterator, indices, features, targets, batch_size):
		x, y, iterator = self.next(
									iterator, 
									indices, 
									features, 
									targets,
									batch_size
								)

		if iterator > len(indices)//batch_size:
			iterator = 0
	
		x, y = self.loader.get_batch(x, y)
		x, y = self.preprocessor.apply(x, y)
		x, y = self.augmentor.apply(x, y)

		return x, y, iterator
	
	'''
		get_train_batch : get next training batch
	'''
	def get_train_batch(self):
		x, y, self.train_iterator = self.get_batch(
			self.train_iterator,
			self.train_indices,
			self.train_features,
			self.train_targets,
			self.config.HYPERPARAMETERS.TRAIN_BATCH_SIZE
		)

		return x, y

	'''
		get_valid_batch : get next validation batch
	'''
	def get_valid_batch(self):
		x, y, self.valid_iterator = self.get_batch(
			self.valid_iterator, 
			self.valid_indices, 
			self.valid_features, 
			self.valid_targets,
			self.config.HYPERPARAMETERS.VALID_BATCH_SIZE
		)

		return x, y
		

	'''
	get_data_info : extracts base information about the data for generating batches and using
					it for getting batches of data from the feeders.
	'''
	# Getting indices and features and targets for the dataset.
	def get_dset_info(self):
		self.interface_type = self.config.DATA.DATASET.INTERFACE_TYPE
		self.split_data = self.config.DATA.SPLIT_DATA
		self.feature_names = self.config.DATA.DATASET.FEATURES
		self.target_names = self.config.DATA.DATASET.TARGETS
			
		# If the dataset info is given as a csv file
		if self.interface_type == 'csv':
			# If dataset is to be split in train and valid sets.
			train_features, valid_features, train_targets, valid_targets = get_features_labels_from_csv()
					
		elif self.interface_type == 'dir':
			train_features, valid_features, train_targets, valid_targets = get_features_labels_from_dirs()
				
		elif self.interface_type == 'text':
			print('text')
		elif self.interface_type == 'json':
			print('json')
		elif self.interface_type == 'xml':
			print('xml')

		self.train_features = train_features
		self.train_targets = train_targets
		self.train_indices = np.arange(0, len(train_features))
		self.valid_features = valid_features
		self.valid_targets = valid_targets
		self.valid_indices = np.arange(0, len(valid_features))