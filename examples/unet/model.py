from pai.data import IMAGE_FEEDER
from pai.settings import CONFIG
from pai.models import MODEL
from pai.trainer import TRAINER

# Keras Or PyTorch Imports To Create Your Model.

config = CONFIG('./config.json')

class UNET(MODEL):
	def compose_model(self):
		# model code
		return model


model = UNET(config)

feeder = IMAGE_FEEDER(config)

trainer = TRAINER(config, model, feeder)

model = trainer.train()

trainer.evaluate()

pred = model.predict(new_image)