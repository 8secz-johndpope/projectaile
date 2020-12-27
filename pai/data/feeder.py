class FEEDER():
	def __init__(self, config, feed_function):
		self.config = config
		self.feed_function = feed_function

	def augment(self):
		# Include Out Of The Box Augmentations For Image, Text, Audio And Structrured Data
		return

	def preprocess(self):
		# Preprocessing Steps To Be Performed
		return

	def feed(self, x, y):
		# 1. read x and y
		# 2. perform preprocess steps
		# 3. run augmentations based on the probability of augmentations
		# 4. return x and y
		pass