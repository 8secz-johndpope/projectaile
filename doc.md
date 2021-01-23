![](https://t3714927.p.clickup-attachments.com/t3714927/7f6abddc-f516-4212-89ab-a12647ac71df/projectaile_banner_no_bg.png)

> A framework agnostic architecture and utility library for all machine learning and deep learning projects providing an abstract and easy to use but still very configurable API.

  

Idea :
======

The idea behind ProjectAile is to remove the frustration of understanding and plugging open source machine learning projects in your own code and to provide a uniform architecture and API for all frameworks and projects making it easier to re-implement or integrate code from research papers or other opensource projects.

  

There's also a common problem of re-writing boiler plate stuff like creating data pipelines, model utility functions, callbacks configurations, etc. ProjectAile provides utility functions and easily extendible classes for removing the hassle of writing the boilerplate stuff.

  

### Features :

ProjectAile aims to provide the following functionalities at it's core :

  

#### \* Configuration is all you need :

All the project configurations and setup through a simple json file that does most of the stuff and you won't have to write any extra code for what is obvious.

  

#### \* Common API for all frameworks :

No more confusion between frameworks, write in whichever you're comfortable and use the same API for all frameworks.

  

#### \* Extendibility :

ProjectAile comes with inbuild support for ladle and is also at the core of dlflow, making creating new projects and using different components more easy. DLFlow provides a command line utility for easily creating and running tasks on ProjectAile projects and deploying them while ladle provides snippets of code or different component implementations like loss functions, optimizers, visualizations etc. for extending ProjectAile.

  

#### \* Support :

ProjectAile supports the most common AI frameworks ( TensorFlow and PyTorch , more to come soon ) and also most common languages for AI Implementations ( Python, C/C++ support to come in future ) thus making it easier to focus on what actually needs to be done and cutting short the development time.

  

#### \* Common project architecture :

It gets difficult to read someone else's code, everyone has their own preferences and conventions, making it a bit difficult to navigate through their code, especially in case of research paper implementations where, finding production level code or even easy to understand code is difficult. ProjectAile provides a common project structure that has well thought out directory structure, that is simple, easy to understand and at the same time provides all that you need for your experiments.

  

#### \* From research to development to deployment :

ProjectAile is built with Machine Learning Research and Development in mind. Thus it is easy to grasp and quick to start with for all the research experiments that require quick experimentation and testing, as well as scalable and clean for production level deployment.

Thus it focuses on the whole ML R&D pipeline and supports going from R to D and to D.

  

### 1\. Configuration is all you need

The config.json is the configuration file of the project and contains several detailed sections that are used internally by ProjectAile for everything from data pipeline to model training to deployment.

  

The configuration file has 4 basic sections :

```json
{
  "MODEL" : {
    // Contains The Model Configurations, like input shape, loss function[s], optimizer, 
    // model name, model type, whether to use a prebuilt model etc.
  },
  "DATA" : {
    // Contains the configuration for the dataset and defining the structure and type of
    // data interface and the directory structure and if its a standard dataset or custom.
  },
  "HYPERPARAMETERS" : {
    // Contains configuration parameters for different hyperparameters used in the model
    // like learing rate, epochs, validation split size, metric to monitor, etc.
  },
  "CONFIG_INFO" : {
    // Contains the configuration parameters related to model saving and evaluation results
    // , logs path, deployment parameters like cloud service, credential file path etc.
  }
}
```

  

Each section contains subsections for defining the project in more detail and customizing as per the requirements.

  

For detailed configuration files, check out the example configuration files [here](https://github.com/explabs-ai/projectaile)

For all the available options/keys that are used by ProjectAile internally, check the configuration [documentation](https://github.com/explabs-ai/projectaile) .

  

One great thing about ProjectAile being at the core of DLFlow is that, you don't have to create this configuration from scratch, you can simply use dlflow commandline utility and answer some questions about the project to generate the configuration which can be edited easily later.

  

You can also add your own sections and keys to the configuration file and they'll be available in the config object under the added section.

  

  

### 2\. Common API for all frameworks

> Working on finalizing this.

  

### 3\. Extendibility