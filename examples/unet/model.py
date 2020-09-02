from projectaile import MODEL, CONFIG, TRAINER
from .feeder import UNET_FEEDER

class UNET(MODEL):
	def __init__(self, config):
		model_name = 'unet'
		super(MODEL, self).__init__(config, model_name)

	def compose_model(self):
		# model code
		return


if __name__ == '__main__':
	config = CONFIG('./config.json')
	model = UNET(config)
	loader = LOADER(config, UNET_FEEDER)
	trainer = TRAINER(config, model, loader)
	trainer.train()