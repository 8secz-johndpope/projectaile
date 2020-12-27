import numpy as np
import pandas as pd
import sklearn

class LOADER:
	def __init__(self, config, feeder=None, mode='train'):
		self.config = config
		self.feeder = feeder
		self.mode = mode
		if self.mode == 'train':
			self.batch_size = self.config.HYPERPARAMETERS['TRAINING_BATCH_SIZE']
		elif self.mode == 'valid':
			self.batch_size = self.config.HYPERPARAMETERS['VALID_BATCH_SIZE']

		self.get_dset()

	def get_dset(self):
		interface = self.config.DATASET['INTERFACE_TYPE']
		interface_path = self.config.DATASET['INTERFACE_PATH']
		is_dir = self.config.DATASET['INTERFACE_DIRECTORIES']

		if interface == 'csv':
			dset = pd.read_csv(interface_path)
			self.features = dset[self.config.DATASET['FEATURE_COLS']]
			self.labels = dset[self.config.DATASET['TARGET_COLS']]

		elif interface == 'directory':
			# Further splitting if self.config.dataset.interface_directories is true
			return

		elif interface == 'txt':
			with open(interface_path, 'r') as f:
				dset = f.read()
				dset = [i for i in dset.split('\n') if len(i) != 0]
				delimiter = self.config.DATASET['DELIMITER']
				x_idx = self.config.DATASET['FEATURE_COLS']
				y_idx = self.config.DATASET['TARGET_COLS']
				self.features = [np.array(i.split(delimiter))[x_idx] for i in dset]
				self.labels = [np.array(i.split(delimiter))[y_idx] for i in dset]

		self.indices = np.arange(len(self.features))

	def shuffle(self):
		assert len(self.features) == len(self.labels)
		p = np.random.permutation(len(self.features))
		self.features = self.features[p]
		self.labels = self.labels[p]

	def load_batches(self):
		num_batches = len(self.indices) // self.batch_size
		self.shuffle()

		for i in range(num_batches):
			start_idx = self.batch_size * i
			end_idx = self.batch_size * (i + 1)
			batch_x, batch_y = self.feeder.feed(self.features[start_idx:end_idx], self.labels[start_idx:end_idx])
			yield batch_x, batch_y

	def split_dset(self):
		self.get_dset()

		interface = self.config.DATASET['INTERFACE_TYPE']

		if interface == 'csv':
			return
		elif interface == 'txt':
			return
		elif interface == 'directory':
			return
		else:
			raise Exception('Not Supported Interface')

		train_x, valid_x, train_y, valid_y = sklearn.model_selection.train_test_split(
			self.features,
			self.labels,
			test_size=self.config.DATASET['TEST_SPLIT_SIZE'],
			random_state=42
		)

		self.train_x = train_x
		self.train_y = train_y
		self.valid_x = valid_x
		self.valid_y = valid_y