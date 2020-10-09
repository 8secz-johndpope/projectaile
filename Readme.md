<img src='src/logo.png' alt='logo' />
A common api architecture and design language for all machine learning related projects with library support in Python ( with tensorflow(keras) and pytorch framework support ) and C++.
The end goal for the project is to create a design framework for machine learning projects to provide uniformity in the vastly non-uniform world of machine learning research and development projects.
The ProjectAIle provides a common project structure, useful functionality through a standard api and a guideline for taking any product from research to production.
After working on several machine learning projects and facing several difficulties in going from a raw research and experimentation based development to production based development, i've constantly optimized my workflow and development process as well as the code organization structure and have created techniques and methods that can be utilized to reduce the boiler plate stuff and takes out the hassle of re-writing or modifying several mundane tasks that can slow down the speed of development.
Although everyone have their own style of development and their own way of organizing their code, through ProjectAIle, i plan to provide a common checklist or pipeline or workflow which is essentially common in every machine learning project and can be easily modified by anyone or extended upon to customize their workflow.

### The Directory Structure
The ProjectAIle architecture contains the following project structure : <br />
<img src='src/directory.png' alt='directory' />


### The API
The aim of the project is to provide helper functions so you've to write minimal code with the likes of the following
```python
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

pred = model.predict(new_image
```
ProjectAIle will provide extendable classes for the following: <br />
#### 1. DATA_LOADER : (Implicit, uses the feeder to pass in the batch information, the feeder does the loading and pre-processing etc.)
#### 2. FEEDER : (Base class for the feeder where you can define the loading process, the pre-process steps and any augmentations)
#### 3. MODEL : (Base class for the model with all the necessary functions like plotting the model architecture, printing the summary, saving the model weights, loading a model etc.)
#### 4. TRAINER : (The trainer that trains the model and provides the evaluation loops as well, returns the trained model)
